"""
    this code is used to extract data from the database and save it in a csv file
"""

import pyodbc
import csv

# Use suas credenciais para se conectar ao banco de dados
"""Test database"""
"""
server = 'jenna-germana.intrajus.tjrn'
database = 'E-guia_teste'
username = 'eguia'
password = 'devsoft@'
"""

"""Production database"""
server = "jenna-germana.intrajus.tjrn"
database = "E_GUIA_DIARIO"
username = "eguia_diario"
#password = "devsoft@"
password = username

driver = '{ODBC Driver 17 for SQL Server}'

# Crie a string de conexão
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    # Crie a conexão
    conn = pyodbc.connect(connection_string)
except pyodbc.Error as e:
    print(f"Failed to connect to the database: {e}")
    exit(1)

# Crie um cursor
cursor = conn.cursor()

# Especifique as tabelas que você quer
#tables = ["ordempagamento", "ItemOrdemPagamento", "OrdemPagamentoExtrajudicial", "fatura"]
#tables = ["OrdemPagamentoExtrajudicial"]
tables = ["OrdemPagamento"]

data = []
column_names_set = set()

# Itere sobre todas as tabelas
for table in tables:
    try:
        # Obtenha todos os registros dessa tabela
        cursor.execute(f"SELECT * FROM {table};")
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        column_names_set.update(column_names)
    except pyodbc.Error as e:
        print(f"Failed to fetch data from table {table}: {e}")
        continue

    # Adicione os registros ao data
    for row in rows:
        record = {name: (str(value) if value is not None else None) for name, value in zip(column_names, row)}
        record["table"] = table
        data.append(record)

# Convertemos o conjunto em uma lista para podermos usá-la como fieldnames no DictWriter
column_names_list = list(column_names_set)
column_names_list.append('table')  # Não esqueça de incluir a coluna 'table'

# Escreva os dados em um arquivo CSV
try:
    with open('ordem_pagamento_producao.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=column_names_list)
        writer.writeheader()
        for record in data:
            writer.writerow({field: record.get(field, None) for field in column_names_list})
except Exception as e:
    print(f"Failed to write data to CSV file: {e}")

# Feche a conexão
conn.close()