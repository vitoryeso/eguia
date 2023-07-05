import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_extrajudicial(df):
    """
    Realiza o pré-processamento dos dados extrajudiciais. 
    Fazendo:
        - Criando a coluna alvo "valor" a partir da soma das colunas "valorFCRCPN", "valorFDJ" e "valorUnitario"
        - Removendo colunas desnecessárias
        - Ajustando a coluna "dataCadastro", removendo os microsegundos e convertendo para datetime
        - Realiza checagens para garantir que o DataFrame está correto
    """

    # Criando a coluna alvo "valor"
    df["valor"] = df[["valorFCRCPN", "valorFDJ", "valorUnitario"]].sum(axis=1)
    
    # Removendo colunas desnecessárias
    columns_to_remove = ["cartorioId", "valorFCRCPN", "valorFDJ", "valorUnitario", "dataFim", 
                         "situacao", "cnpj", "dataInicio", "emailDeEnvio", "tipoServico_id", 
                         "id", "table"]
    df = df.drop(columns=columns_to_remove)
    
    # Ajustando a coluna "dataCadastro"
    df['dataCadastro'] = df['dataCadastro'].str.slice(0, 19) # remove os microsegundos
    df["dataCadastro"] = pd.to_datetime(df["dataCadastro"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    
    # Checando se sobraram apenas as colunas "nome", "valor", e "dataCadastro"
    assert set(df.columns) == {"nome", "valor", "dataCadastro", "quantidadeDeSelos"}

    # Checando se não existem valores faltantes
    assert not df.isna().values.any(), "Existem valores faltantes no DataFrame."

    return df

def preprocess_intrajudicial(df, orgao_julgador_df):
    """
    Realiza o pré-processamento dos dados intrajudiciais.
    Fazendo:
        - Mapeando os valores da coluna id_orgaojulgador para o nome correspondente, usando o DataFrame orgao_julgador_df
        - Preenchendo os valores faltantes da coluna "id_orgaojulgador" com "Desconhecido"
        - Ajustando a coluna "dataCadastro", removendo os microsegundos e convertendo para datetime
        - Removendo colunas desnecessárias
        - Realiza checagens para garantir que o DataFrame está correto

    """

    # Criando um dicionário com o id_orgaojulgador como chave e o nome correspondente como valor
    orgao_julgador_map = orgao_julgador_df.set_index('id')['nome'].to_dict()

    # Mapeando os valores da coluna id_orgaojulgador para o nome correspondente
    df['id_orgaojulgador'] = df['id_orgaojulgador'].map(orgao_julgador_map)

    # Renomeando a coluna "id_orgaojulgador" para "orgao_julgador"
    df.rename(columns={'id_orgaojulgador': 'nome'}, inplace=True)

    # Preenchendo os valores faltantes da coluna "id_orgaojulgador" com "Desconhecido"
    df['nome'] = df['nome'].fillna('Desconhecido')

    # Ajustando a coluna "dataCadastro"
    df['dataCadastro'] = df['dataCadastro'].str.slice(0, 19) # remove os microsegundos
    df["dataCadastro"] = pd.to_datetime(df["dataCadastro"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    
    # Removendo colunas desnecessárias
    columns_to_remove = ["informacaoComplementar", "idUnidade", "desconto", "justicaGratuita", 
                         "valorCobrado", "situacao", "id", "table"]
    df = df.drop(columns=columns_to_remove)
    
    # Checando se não existem valores faltantes
    assert not df.isna().values.any(), "Existem valores faltantes no DataFrame."

    return df

def concat_dataframes(df_intrajudicial, df_extrajudicial):
    """
    Concatena os dataframes de ordem de pagamento intrajudicial e extrajudicial. 
    """

    # Adicionando as colunas 'intrajudicial' e 'extrajudicial'
    df_intrajudicial['intrajudicial'] = 1
    df_intrajudicial['extrajudicial'] = 0
    df_extrajudicial['intrajudicial'] = 0
    df_extrajudicial['extrajudicial'] = 1

    # Concatenando os dataframes
    df = pd.concat([df_intrajudicial, df_extrajudicial], ignore_index=True)
    df['quantidadeDeSelos'] = df['quantidadeDeSelos'].fillna(0.0)

    # Ordenando o dataframe pela coluna 'dataCadastro'
    df = df.sort_values(by=['dataCadastro'])

    return df

def destilate_data(df):
    """
    Extrai as informações de data e hora da coluna 'dataCadastro' e as coloca em colunas separadas. 
    """

    df['dataCadastro'] = pd.to_datetime(df['dataCadastro'])
    df['ano'] = df['dataCadastro'].dt.year
    df['mes'] = df['dataCadastro'].dt.month
    df['dia'] = df['dataCadastro'].dt.day
    df['hora'] = df['dataCadastro'].dt.hour
    df['minuto'] = df['dataCadastro'].dt.minute
    df['segundo'] = df['dataCadastro'].dt.second

    # Após isso, podemos remover a coluna original 'dataCadastro'
    df = df.drop(['dataCadastro'], axis=1)

    return df

def tokenize_nomes(df):
    """
    Realiza a tokenização dos nomes das ordens de pagamento. (comarcas e cartórios), utililzando o CountVectorizer.
    """

    vectorizer = CountVectorizer()

    # Ajusta o vetorizador no texto 'nome'
    X = vectorizer.fit_transform(df['nome'])

    # Insere as novas colunas no DataFrame
    df = pd.concat([df, pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())], axis=1)

    # Agora podemos remover a coluna original 'nome'
    df = df.drop(['nome'], axis=1)

    return df

def main():
    # Carregando e pré-processando o dataframe de ordem de pagamento extrajudicial
    df_extrajudicial = pd.read_csv("data/raw/ordem_pagamento_extrajudicial_producao.csv")
    df_extrajudicial = preprocess_extrajudicial(df_extrajudicial)
    df_extrajudicial.to_csv("data/processed/processed_ordem_pagamento_extrajudicial_producao.csv", index=False)

    # Carregando e pré-processando o dataframe de ordem de pagamento intrajudicial
    df_intrajudicial = pd.read_csv("data/raw/ordem_pagamento_producao.csv")
    df_orgao_julgador = pd.read_csv("data/raw/orgao_julgador.csv") # Carregando o dataframe de orgão julgador
    df_intrajudicial = preprocess_intrajudicial(df_intrajudicial, df_orgao_julgador)
    df_intrajudicial.to_csv("data/processed/processed_ordem_pagamento_intrajudicial_producao.csv", index=False)

    # Concatenando os dataframes
    df = concat_dataframes(df_intrajudicial, df_extrajudicial)
    df = destilate_data(df)
    df = tokenize_nomes(df)
    # Realizando casting pra float32
    df = df.astype(float)
    print(df)

    df.to_csv("data/processed/processed_ordem_pagamento_producao.csv", index=False)


if __name__ == "__main__":
    main()
