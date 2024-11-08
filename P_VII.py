import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import levene, ttest_ind


visits_df = pd.read_csv('./visits_log_us.csv')
orders_df = pd.read_csv('./orders_log_us.csv')
costs_df = pd.read_csv('./costs_us.csv')

print('En este proyecto analizaremos tres datasets, correspondientes a visitas, ordenes y costos, analizaremos el comportamiento de los usuarios junto con sus metricas, lo cual nos ayudará a tomar decisiones sobre como podemos aumentar ventas, a que áreas podremos aumentar o reducir el presupuesto para minimizar o erradicar perdidas y aumentar ventas lo cual es ingreso para la empresa.')
print()

print('Preparacion de datos')
print()

print(visits_df.info())
print()

visits_df['start_session'] = pd.to_datetime(visits_df['Start Ts'])
visits_df['end_session'] = pd.to_datetime(visits_df['End Ts'])
visits_df.columns = [ columna.lower().replace(' ','_') for columna in visits_df.columns]

print(visits_df.head())
print()

visits_duplicated = visits_df.duplicated().sum()
print(f'Filas duplicadas : {visits_duplicated}')
print()

visits_null = visits_df.isna().sum()
print(f'Valores nulos : \n{visits_null}')
print()

print(visits_df.value_counts('device'))
print()

print('Mediante la siguiente información podemos observar que hay 262 mil visitas en computadoras y casi 97 mil visitas en dispositivos touch, lo cual nos indica que hay mucha mas actividad mediante un ordenador que un dispositivo touch o movil.')
print()

print(visits_df.value_counts('source_id'))
print()

print('Observamos que tenemos 10 fuentes, la cual la mayor tiene casi 102 mil visitas, de que plataforma se trata, no sabemos pero podemos averiguar pidiendo mas indformación de referencia.')
print()

print(visits_df['start_session'].min())
print(visits_df['start_session'].max())
print()

print(visits_df['end_session'].min())
print(visits_df['end_session'].max())
print()

print(orders_df.info())
print()

orders_df['buy_tsn'] = pd.to_datetime(orders_df['Buy Ts'])
orders_df.columns = [ columna.lower().replace(' ','_') for columna in orders_df.columns]
print(orders_df.head())
print()

orders_duplicated = orders_df.duplicated().sum()
print(f'Filas duplicadas : {orders_duplicated}')
print()

orders_null = orders_df.isna().sum()
print(f'Valores nulos : \n{orders_null}')
print()

print(costs_df.info())
print()

costs_df['dt'] = pd.to_datetime(costs_df['dt'], format="%Y-%m-%d")
costs_df.columns = [ columna.lower().replace(' ','_') for columna in costs_df.columns]
print(costs_df.head())
print()

costs_duplicated = costs_df.duplicated().sum()
print(f'Filas duplicadas : {costs_duplicated}')
print()

costs_null = costs_df.isna().sum()
print(f'Valores nulos : \n{costs_null}')
print()

print('1.- VISITAS')
print()

print('Usuarios por día, semana y mes.')
print()

visits_df['session_year'] = visits_df['start_session'].dt.isocalendar().year
visits_df['session_month'] = visits_df['start_session'].dt.month
visits_df['session_week'] = visits_df['start_session'].dt.isocalendar().week
visits_df['session_date'] = visits_df['start_session'].dt.date

print(visits_df.head())
print()

print('Cantidad de usuarios unicos que visitan la pagina web')
print()

users_day = visits_df.groupby('session_date')['uid'].nunique()
users_week = visits_df.groupby('session_week')['uid'].nunique()
users_month = visits_df.groupby('session_month')['uid'].nunique()

fig, ax = plt.subplots(1,3, figsize=(19,5))
ax[0].plot(users_day)
ax[0].set(title = 'Usuarios día', xlabel='Fecha', ylabel='Visitas')
ax[1].plot(users_week)
ax[1].set(title = 'Usuarios semana')
ax[1].set(title = 'Usuarios semana', xlabel='Fecha', ylabel='Visitas')
ax[2].plot(users_month)
ax[2].set(title = 'Usuarios mes')
ax[2].set(title = 'Usuarios mes', xlabel='Fecha', ylabel='Visitas')
fig.autofmt_xdate(rotation=30)

print('Cantidad de sesiones por día (Un usuario puede tener mas de una sesion')
print()

users_sessions = visits_df.groupby('session_date').agg({'uid' : ['count','nunique']})
users_sessions.columns = ['sessions_num', 'users_num']
users_sessions['session_per_user'] = users_sessions['sessions_num'] / users_sessions['users_num']
print(users_sessions)
print()

plt.figure(figsize=(12,5))
users_sessions['session_per_user'].plot().set(title='Sesiones por usuario', xlabel='Fecha', ylabel='Sesiones')
plt.show()

print('El promedio de sesiones por ususario es:', round(users_sessions['session_per_user'].mean(),2))
print()

visits_df['session_seg'] = (visits_df['end_session'] - visits_df['start_session']).dt.seconds
visits_df['session_min'] = (visits_df['session_seg']) / 60
print(visits_df.describe())
print()

plt.figure(figsize=(12,5))
visits_df['session_seg'].hist(bins=300).set(ylabel='Numero sesiones', xlabel='Duracion')
plt.xlim(0,10000)
plt.show()

print('Promedio de duarcion de sesion [minutos]:', visits_df['session_seg'].mean()/60)
print()

print('Mediana de duarcion de sesion [minutos]:', visits_df['session_seg'].median()/60)
print()

print('Moda de duarcion de sesion [minutos]:', visits_df['session_seg'].mode()[0]/60)
print()

dau_total = visits_df.groupby('session_date').agg({'uid': 'nunique'}).mean()
wau_total = visits_df.groupby(['session_year', 'session_week']).agg({'uid':'nunique'}).mean()
mau_total = visits_df.groupby(['session_year', 'session_month']).agg({'uid':'nunique'}).mean()

print(f'Actividad por dia : {int(dau_total)}')
print()
print(f'Actividad por semana : {int(wau_total)}')
print()
print(f'Actividad por mes : {int(mau_total)}')
print()

print('Factor de adherencia, calcularemos con que frecuencia regresan los usuarios a la página.')
print()

sticky_wau = dau_total / wau_total * 100
print(f'Frecuencia semanal de retorno: {int(sticky_wau)}')
print()
sticky_mau = dau_total / mau_total * 100
print(f'Frecuencia  mensual de retorno: {int(sticky_mau)}')
print()

print('2.- VENTAS')
print()

first_order_date_by_uid = orders_df.groupby('uid')['buy_tsn'].min()
first_order_date_by_uid.name = 'first_order_date'
orders_df = orders_df.join(first_order_date_by_uid, on = 'uid')
orders_df['first_order_month'] = pd.to_datetime(orders_df['first_order_date'], format="%Y-%m-%d")
orders_df['order_month'] = pd.to_datetime(orders_df['buy_tsn'], format="%Y-%m-%d")
print(orders_df.head())
print()

print('Elaboracion de Tabla dinamica para observar la tendencia de comportamiento de nuestros clientes en cuanto a compras desde que inician una sesion.')
print()

orders_df.pivot_table(
    index = 'first_order_month',
    columns = 'order_month',
    values = 'uid',
    aggfunc = 'nunique')

print('Vamos a encontrar el ciclo de vida de la cohorte. Restaremos el mes de la cohorte (first_order_month) del mes en que se realizaron las compras (order_month):')
print()

orders_grouped_by_cohorts = orders_df.groupby(['first_order_month', 'order_month']).agg({'revenue':'sum', 'uid':'nunique'})
orders_grouped_by_cohorts = orders_grouped_by_cohorts.reset_index()
orders_grouped_by_cohorts['cohort_lifetime'] = (orders_grouped_by_cohorts['order_month']- orders_grouped_by_cohorts['first_order_month'])
print(orders_grouped_by_cohorts.head())
print()

print('Ahora enocontraremos el ingreso por usuario')
print()

orders_grouped_by_cohorts['revenue_per_user'] = (orders_grouped_by_cohorts['revenue'] / orders_grouped_by_cohorts['uid'])

print('Ahora trazaremos una tabla dinámica que muestre los cambios en los ingresos por usuario para las cohortes por mes de compra y evaluaremos los cambios en los ingresos por usuario a lo largo del tiempo:')
print()

orders_grouped_by_cohorts.pivot_table(
    index='first_order_month',
    columns='order_month',
    values='revenue_per_user',
    aggfunc='mean',
)

orders_grouped_by_cohorts['cohort_lifetime'] = orders_grouped_by_cohorts['cohort_lifetime'] / np.timedelta64(1, 'M')

orders_grouped_by_cohorts['cohort_lifetime'] = (orders_grouped_by_cohorts['cohort_lifetime'].round().astype('int'))

print(orders_grouped_by_cohorts[['first_order_month', 'order_month', 'cohort_lifetime']].head())
print(orders_grouped_by_cohorts[['first_order_month', 'order_month', 'cohort_lifetime']].tail())
print()

print('Ahora tenemos números enteros de meses. Nos dicen el número del mes de compra relativo al mes de la cohorte.')
print()

orders_grouped_by_cohorts['first_order_month'].dt.strftime('%d.%m.%Y')
orders_grouped_by_cohorts['first_order_month'] = orders_grouped_by_cohorts['first_order_month'].dt.strftime('%Y-%m')

print('Ahora compilaremos una tabla dinámica de los cambios en el ingreso por usuario, cuyas columnas contendrán el ciclo de vida y cuyas filas serán las cohortes.')
print()

revenue_per_user_pivot = orders_grouped_by_cohorts.pivot_table(
    index='first_order_month',
    columns='cohort_lifetime',
    values='revenue_per_user',
    aggfunc='mean',
)
print(revenue_per_user_pivot)
print()

plt.figure(figsize=(13, 9))
plt.title('Tamaño promedio de compra del cliente')
sns.heatmap(
    revenue_per_user_pivot,
    annot=True,
    fmt='.1f',
    linewidths=1,
    linecolor='black',
)

orders_df['order_date'] = pd.to_datetime(orders_df['buy_tsn'])
costs_df['date'] = pd.to_datetime(costs_df['dt'])

orders_df['order_month2'] = orders_df['order_date'].astype('datetime64[M]')
costs_df['month'] = costs_df['date'].astype('datetime64[M]')

first_orders = orders_df.groupby('uid').agg({'order_month': 'min'}).reset_index()
first_orders.columns = ['uid', 'first_order_month1']
print(first_orders.head())
print()

cohort_sizes = (first_orders.groupby('first_order_month1').agg({'uid': 'nunique'}).reset_index())
cohort_sizes.columns = ['first_order_month', 'n_buyers']

print(cohort_sizes.head())
print()

orders_ = pd.merge(orders_df,first_orders, on='uid')
print(orders_.head())
print()

cohorts = orders_.groupby(['first_order_month','order_month']).agg({'revenue': 'sum'}).reset_index()
print(cohorts.head())
print()

report = pd.merge(cohort_sizes, cohorts, on='first_order_month')
print(report.head())
print()

margin_rate = 0.5
report['gp'] = report['revenue'] * margin_rate
report['age'] = (report['order_month'] - report['first_order_month']) / np.timedelta64(1, 'M')
report['age'] = report['age'].round().astype('int')
print(report.head())
print()

report['ltv'] = report['gp'] / report['n_buyers']

output = report.pivot_table(
    index='first_order_month', 
    columns='age', 
    values='ltv', 
    aggfunc='mean').round()
output.fillna('')

monthly_costs = costs_df.groupby('month').sum()
report_ = pd.merge(report, monthly_costs, left_on='first_order_month', right_on='month')
report_['cac'] = report_['costs'] / report_['n_buyers']

report_['romi'] = report_['ltv'] / report_['cac']
result = report_.pivot_table(index='first_order_month', columns='age', values='romi', aggfunc='mean')

result = result.fillna('')

monthly_costs = costs_df.groupby('month').sum()
report_ = pd.merge(report, monthly_costs, left_on='first_order_month', right_on='month')
report_['cac'] = report_['costs'] / report_['n_buyers']

report_['romi'] = report_['ltv'] / report_['cac']
result = report_.pivot_table(index='first_order_month', columns='age', values='romi', aggfunc='mean')


result = report_.pivot_table(index='first_order_month', columns='age', values='ltv', aggfunc='mean')

m6_cum_ltv = result.cumsum(axis=1).mean(axis=0)[5]

print('El LTV promedio durante 6 meses desde el primer pedido:', m6_cum_ltv)
print()

print('3.- MARKETING')
print()

print('Para obtener un resumen del gasto total por fuente de adquisición a lo largo del tiempo, primero necesitamos asegurarnos de que nuestro conjunto de datos de costos (costs_df) incluya una columna que identifique la fuente de adquisición. Si esa columna no está presente, no podremos segmentar correctamente.')
print()

total_spent = costs_df['costs'].sum()
print(f'Total gastado: {total_spent}')

spent_by_source = costs_df.groupby('source_id')['costs'].sum().reset_index()

monthly_spent = costs_df.groupby('month')['costs'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='costs', data=monthly_spent)
plt.title('Gasto a lo largo del tiempo')
plt.xlabel('Mes')
plt.ylabel('Gasto Total')
plt.xticks(rotation=45)
plt.show()

print('Para calcular el CAC por fuente, mira la relación entre los costos por fuente y la cantidad de compradores únicos adquiridos a partir de esa fuente.')
print()

report_ = pd.merge(report, monthly_costs, left_on='first_order_month', right_on='month')
report_['cac'] = report_['costs'] / report_['n_buyers']

report_['romi'] = report_['ltv'] / report_['cac']
result = report_.pivot_table(index='first_order_month', columns='age', values='romi', aggfunc='mean')

print(report_)
print()

