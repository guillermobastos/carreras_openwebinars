import pandas as pd 

df1 = pd.read_excel('C:/Users/Usuario/Documents/trabajo/Tabla1.xlsx')
df2 = pd.read_excel('C:/Users/Usuario/Documents/trabajo/Tabla2.xlsx')


df_merged = pd.merge(df1, df2, on=['Dia', 'Mes', 'Ano'])
df_sin_duplicados = df_merged.drop_duplicates(subset=['Dia','Mes','Ano'])
df_sin_duplicados.to_excel('C:/Users/Usuario/Documents/trabajo/Tabla_Final_Sin_Duplicados.xlsx',index=False)