"""Preparação de dados para produção — Airbnb Amsterdam.

Transforma a base crua do InsideAirbnb (96 colunas) nas 27 features que o
modelo espera, reproduzindo exatamente o pipeline validado no notebook
(Etapas 02, 03 e 06).

REGRA CENTRAL: esta classe NUNCA aprende nada dos dados que recebe.
Todo parâmetro aprendido vem do dicionário `artifacts`, gerado no treino.
É a diferença entre `fit_transform` e `transform` — e é o que garante que
produção reproduza o que foi validado (MAPE de 22,21%).

Plano: docs/plans/13_deploy_producao.md §3
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes — NÃO são aprendidas dos dados (plano §2.2)
#
# Recalculá-las daria o mesmo resultado; ficam aqui por serem parte da
# definição das features, não parâmetros estimados.
# ---------------------------------------------------------------------------

COLS_MONETARIAS = ['price', 'security_deposit', 'cleaning_fee', 'extra_people']

COLS_FLAGS = [
    'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    'is_location_exact', 'instant_bookable', 'require_guest_profile_picture',
    'require_guest_phone_verification',
]

FLAGS_AMENITIES = {
    'tem_tv': ['TV'],
    'tem_washer': ['Washer'],
    'tem_dryer': ['Dryer'],
    'tem_family_kid_friendly': ['Family/kid friendly'],
    'tem_cable_tv': ['Cable TV'],
    'tem_private_entrance': ['Private entrance'],
    'tem_dishwasher': ['Dishwasher'],
    'tem_patio_balcony': ['Patio or balcony'],
    'tem_free_parking': ['Free parking on premises', 'Free street parking'],
    'tem_elevator': ['Elevator', 'Elevator in building'],
    'tem_step_free_access': ['Step-free access'],
}

SUBNOTAS = [
    'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location', 'review_scores_value',
]

CAMPOS_TEXTO = [
    'summary', 'space', 'description', 'neighborhood_overview', 'notes',
    'transit', 'access', 'interaction', 'house_rules', 'host_about',
]

# As 4 regras de validade da Etapa 02. Em produção MARCAM a linha em vez de
# removê-la, para preservar a correspondência 1:1 entrada/saída (plano §3.3).
REGRAS_VALIDADE = [
    ('bathrooms > 10', lambda d: d['bathrooms'] > 10),
    ('minimum_nights > 365', lambda d: d['minimum_nights'] > 365),
    ('maximum_nights == 9999', lambda d: d['maximum_nights'] == 9999),
    ('price <= 0', lambda d: d['price'] <= 0),
]

# Colunas cruas sem as quais o pipeline não roda.
COLUNAS_OBRIGATORIAS = [
    'id', 'price', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'latitude', 'longitude', 'room_type', 'property_type',
    'neighbourhood_cleansed', 'minimum_nights', 'maximum_nights',
    'amenities', 'host_since', 'calculated_host_listings_count',
    'number_of_reviews', 'cleaning_fee', 'security_deposit', 'extra_people',
    'calendar_updated', 'first_review', 'last_review', 'house_rules',
    'description',
]

STATUS_OK = 'ok'
STATUS_EXTRAPOLACAO = 'ok: data posterior a DATA_REF (extrapolacao)'


class ColunasAusentesError(ValueError):
    """Entrada sem colunas obrigatórias. Nomeia quais, em vez de KeyError."""


class DataPreparation:
    """Converte anúncios crus do InsideAirbnb nas 27 features do modelo.

    Parameters
    ----------
    artifacts
        Dicionário gerado por `train.py`, com os parâmetros aprendidos no
        treino: `features`, `categorias_mantidas`, `medianas_mcar`,
        `mapas_ordinal`, `data_referencia`, `polos_geograficos` e os blocos
        de imputação.

    Examples
    --------
    >>> import joblib
    >>> artifacts = joblib.load('app/models/artifacts.pkl')
    >>> prep = DataPreparation(artifacts)
    >>> X = prep.transform(df_cru)          # (n, 27), numérico, sem nulos
    >>> status = prep.ultimo_status          # (n,) — 'ok' ou 'invalido: ...'
    """

    def __init__(self, artifacts: dict):
        self.artifacts = artifacts
        self.features = list(artifacts['features'])
        self.categorias_mantidas = set(artifacts['categorias_mantidas'])
        self.medianas_mcar = dict(artifacts['medianas_mcar'])
        self.mapas_ordinal = {k: dict(v) for k, v in artifacts['mapas_ordinal'].items()}
        self.cols_mnar_zero = list(artifacts['cols_mnar_zero'])
        self.cols_mar_sentinela = list(artifacts['cols_mar_sentinela'])
        self.cols_mcar = list(artifacts['cols_mcar'])
        self.data_referencia = pd.Timestamp(artifacts['data_referencia'])
        self.polos = dict(artifacts['polos_geograficos'])

        # Preenchido por transform(); um rótulo por linha de entrada.
        self.ultimo_status: pd.Series | None = None

    # -- validação ----------------------------------------------------------

    def validar_colunas(self, df: pd.DataFrame) -> None:
        """Falha cedo e explícito se faltar coluna obrigatória (plano §3.4)."""
        faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
        if faltando:
            raise ColunasAusentesError(
                f'Colunas obrigatórias ausentes ({len(faltando)}): '
                f'{", ".join(sorted(faltando))}'
            )

    # -- Etapa 02 -----------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpeza da Etapa 02: parse monetário, flags f/t, marcação de validade.

        Diferente do notebook, NÃO remove linhas: marca em `_status`
        (plano §3.3). A correspondência 1:1 entre entrada e saída é o que
        permite ao cliente não precisar reconciliar índices.
        """
        d = df.copy()

        # Colunas monetárias: "$150.00" -> 150.0 (já pode vir numérico da API)
        for col in COLS_MONETARIAS:
            if col in d.columns and d[col].dtype == object:
                d[col] = (d[col].astype(str)
                          .str.replace(r'[\$,]', '', regex=True)
                          .replace({'nan': np.nan, 'None': np.nan})
                          .astype(float))

        # Flags f/t -> 0/1 (idem: pode já vir 0/1)
        for col in COLS_FLAGS:
            if col in d.columns and d[col].dtype == object:
                d[col] = d[col].map({'f': 0, 't': 1})

        # As 4 regras de validade — marcam, não removem
        status = pd.Series(STATUS_OK, index=d.index, dtype=object)
        for descricao, regra in REGRAS_VALIDADE:
            try:
                violou = regra(d).fillna(False)
            except KeyError:
                continue
            # Preserva o primeiro motivo: uma linha inválida por duas regras
            # reporta a primeira, não a última.
            novo = violou & (status == STATUS_OK)
            status.loc[novo] = f'invalido: {descricao}'

        d['_status'] = status
        return d

    # -- Etapa 03 -----------------------------------------------------------

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        """Distância em km entre dois pontos. Idêntica à célula 49."""
        raio = 6371.0
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return raio * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _parse_calendar_updated(texto):
        """Texto relativo ("3 weeks ago") -> dias. Idêntica à célula 48."""
        if pd.isna(texto):
            return np.nan
        t = str(texto).lower().strip()
        if t == 'today':
            return 0
        if t == 'yesterday':
            return 1
        if t in ('a week ago', 'one week ago'):
            return 7
        if 'never' in t:
            return np.nan
        m = re.match(r'(\d+)\s+(day|week|month)s?\s+ago', t)
        if m:
            n, unidade = int(m.group(1)), m.group(2)
            return n * {'day': 1, 'week': 7, 'month': 30}[unidade]
        return np.nan

    @staticmethod
    def _parse_amenities(texto):
        """Extrai o conjunto de amenities de '{A,"B C",D}'. Idêntica à cél. 50."""
        if pd.isna(texto):
            return set()
        itens = re.findall(r'"([^"]+)"|([^{},]+)', str(texto))
        return {(a or b).strip() for a, b in itens if (a or b).strip()}

    @staticmethod
    def _parse_verificacoes(texto):
        """Extrai verificações de "['email', 'phone']". Idêntica à célula 51."""
        if pd.isna(texto):
            return []
        return re.findall(r"'([^']+)'", str(texto))

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """As features da Etapa 03 que alimentam as 27 colunas finais.

        Todas são determinísticas por linha, EXCETO `property_type_agrupado`,
        que usa `categorias_mantidas` aprendida no treino (plano §3.2).
        """
        d = df.copy()
        ref = self.data_referencia

        # --- Bloco C: temporais (contra DATA_REF fixa, plano §2.2) ---------
        for col in ['host_since', 'first_review', 'last_review']:
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], errors='coerce')
        d['host_experiencia_dias'] = (ref - d['host_since']).dt.days
        d['dias_desde_primeira_review'] = (ref - d['first_review']).dt.days
        d['dias_desde_ultima_review'] = (ref - d['last_review']).dt.days
        d['anuncio_ativo'] = (d['dias_desde_ultima_review'] <= 180).fillna(False).astype(int)
        d['calendar_atualizado_dias'] = d['calendar_updated'].apply(
            self._parse_calendar_updated)

        # --- Bloco A: geográficas ------------------------------------------
        for nome, (lat, lon) in self.polos.items():
            d[nome] = self._haversine_km(d['latitude'], d['longitude'], lat, lon)

        # --- Bloco B: amenities --------------------------------------------
        amen = d['amenities'].apply(self._parse_amenities)
        d['n_amenities'] = amen.apply(len)
        for flag, termos in FLAGS_AMENITIES.items():
            d[flag] = amen.apply(lambda s, t=termos: int(any(x in s for x in t)))

        # --- Bloco D: host --------------------------------------------------
        if 'host_verifications' in d.columns:
            verif = d['host_verifications'].apply(self._parse_verificacoes)
            d['n_host_verifications'] = verif.apply(len)
            d['host_tem_id_verificado'] = verif.apply(
                lambda lst: int('government_id' in lst
                                or 'offline_government_id' in lst))
        d['host_eh_profissional'] = (d['calculated_host_listings_count'] > 1).astype(int)
        if 'host_location' in d.columns:
            d['host_eh_local'] = (d['host_location'].fillna('').str.lower()
                                  .str.contains('amsterdam').astype(int))

        # --- Bloco E: razões estruturais ------------------------------------
        bedrooms = d['bedrooms'].replace(0, np.nan)
        beds = d['beds'].replace(0, np.nan)
        accommodates = d['accommodates'].replace(0, np.nan)
        d['pessoas_por_quarto'] = d['accommodates'] / bedrooms
        d['pessoas_por_cama'] = d['accommodates'] / beds
        d['banheiros_por_pessoa'] = d['bathrooms'] / accommodates
        d['camas_por_quarto'] = d['beds'] / bedrooms
        d['tem_multiplos_banheiros'] = (d['bathrooms'] > 1).astype(int)

        # --- Bloco F: política de reserva -----------------------------------
        d['sem_limite_maximo'] = (d['maximum_nights'] >= 1125).astype(int)
        d['maximum_nights_cap'] = d['maximum_nights'].clip(upper=365)
        d['permite_estadia_curta'] = (d['minimum_nights'] <= 3).astype(int)
        d['taxa_limpeza_por_noite'] = (d['cleaning_fee']
                                       / d['minimum_nights'].replace(0, np.nan))
        d['tem_taxa_limpeza'] = (d['cleaning_fee'].fillna(0) > 0).astype(int)
        d['tem_deposito'] = (d['security_deposit'].fillna(0) > 0).astype(int)

        # --- Bloco G: reviews ------------------------------------------------
        d['sem_reviews'] = (d['number_of_reviews'] == 0).astype(int)
        subnotas_presentes = [c for c in SUBNOTAS if c in d.columns]
        if subnotas_presentes:
            d['review_score_medio'] = d[subnotas_presentes].mean(axis=1)
            if 'review_scores_location' in d.columns:
                d['gap_location_vs_geral'] = (d['review_scores_location']
                                              - d['review_score_medio'])
            if 'review_scores_value' in d.columns:
                d['gap_value_vs_geral'] = (d['review_scores_value']
                                           - d['review_score_medio'])
        meses_ativo = (d['dias_desde_primeira_review'] / 30).replace(0, np.nan)
        d['reviews_por_mes_ativo'] = d['number_of_reviews'] / meses_ativo

        # --- Bloco H: texto ---------------------------------------------------
        d['len_description'] = d['description'].fillna('').astype(str).str.len()
        d['len_house_rules'] = d['house_rules'].fillna('').astype(str).str.len()
        campos_presentes = [c for c in CAMPOS_TEXTO if c in d.columns]
        d['n_campos_texto_preenchidos'] = d[campos_presentes].notna().sum(axis=1)

        # --- property_type_agrupado: USA O PARÂMETRO DO TREINO ---------------
        # Recalcular o corte de 1% sobre o lote recebido daria níveis
        # diferentes e desalinharia o encoding (plano §2.1).
        d['property_type_agrupado'] = d['property_type'].where(
            d['property_type'].isin(self.categorias_mantidas), 'Other')

        # --- Sinaliza extrapolação temporal (plano §2.2) ----------------------
        if '_status' in d.columns:
            posterior = pd.Series(False, index=d.index)
            for col in ['host_since', 'first_review', 'last_review']:
                if col in d.columns:
                    posterior |= (d[col] > ref).fillna(False)
            marcar = posterior & (d['_status'] == STATUS_OK)
            d.loc[marcar, '_status'] = STATUS_EXTRAPOLACAO

        return d

    # -- Etapa 06 -----------------------------------------------------------

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputação por mecanismo + encoding ordinal (Etapa 06, caminho árvore).

        MNAR -> 0 e MAR -> -1 são constantes (derivadas da Análise 3.4).
        As medianas MCAR e os mapas ordinais vêm do treino.
        """
        X = df.copy()

        # MNAR: o nulo significa "não cobro", não "desconhecido" (E06-3)
        for c in self.cols_mnar_zero:
            if c in X.columns:
                X[c] = X[c].fillna(0.0)

        # MAR: sentinela separável por um split; não inventa score (E06-2)
        for c in self.cols_mar_sentinela:
            if c in X.columns:
                X[c] = X[c].fillna(-1.0)

        # Categóricas -> ordinal, com o mapa do TREINO.
        # Categoria nunca vista -> -1 (mesma regra da Etapa 09).
        for c, mapa in self.mapas_ordinal.items():
            if c in X.columns:
                X[c] = X[c].map(mapa).fillna(-1).astype(float)

        # MCAR: mediana do treino
        for c, mediana in self.medianas_mcar.items():
            if c in X.columns:
                X[c] = X[c].fillna(mediana)

        faltando = [c for c in self.features if c not in X.columns]
        if faltando:
            raise ColunasAusentesError(
                f'Features ausentes após o pipeline ({len(faltando)}): '
                f'{", ".join(faltando)}'
            )

        # Ordem EXATA do treino — LightGBM é sensível a ela (plano §2.1)
        return X[self.features].astype(float)

    # -- orquestração -------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline completo: cru -> 27 features numéricas, sem nulos.

        Guarda os rótulos de validade em `self.ultimo_status`, alinhados ao
        índice da entrada.
        """
        self.validar_colunas(df)
        d = self.clean(df)
        d = self.feature_engineering(d)
        self.ultimo_status = d['_status'].copy()
        X = self.prepare(d.drop(columns=['_status']))

        n_nulos = int(X.isna().sum().sum())
        if n_nulos:
            colunas = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f'{n_nulos} nulos restaram após a preparação, em: {colunas}. '
                'Isso indica coluna fora dos blocos de imputação.'
            )
        return X
