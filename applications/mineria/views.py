# Django
from django.shortcuts import render

# Lib
import pyodbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import seaborn as sns
import base64
from io import BytesIO


str_conexion = ("Driver={ODBC Driver 17 for SQL Server};"
                "Server=ANDRES\SQLEXPRESS;"
                "Database=Ferredeposito;"
                "UID=ferre_admin;"
                "PWD=WL27yeJ2ggaxkws6")

conn = pyodbc.connect(str_conexion)


def home(request):

    """--------------------GRAFICA VENTAS POR MES--------------------"""

    query_str = """
        SELECT MONTH(fecha) mes, SUM(valor_total) total_mes FROM ventas_venta
        group by MONTH(fecha) order by 1
    """

    data = pd.read_sql(query_str, conn)

    X = np.array(data[['mes', 'total_mes']])
    fig = plt.figure(figsize=(10, 6))
    plt.plot(X[:, 0], X[:, 1])
    plt.xlabel('Valor ventas')
    plt.ylabel('mes')
    plt.title('Ventas x Mes 2021')

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    img_ventas = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    """--------------------TABLA VENTAS POR MES--------------------"""

    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

    tabla = ''

    for i in range(len(data)):
        tabla += '<tr><td>' + \
            meses[i] + '</td><td>' + \
            '${:,.2f}'.format(data.iloc[i]['total_mes']) + '</td></tr>'

    """--------------------TABLA TOP CLIENTES EN TOTAL--------------------"""

    query_str = """
        SELECT TOP 10 cliente_id, razon_social, SUM(valor_total) as venta FROM ventas_venta ven
        INNER JOIN ventas_cliente cli ON cli.id = ven.cliente_id
        WHERE razon_social != ' PUBLICO GENERAL '
        GROUP BY cliente_id, razon_social
        ORDER BY 3 DESC
    """

    data1 = pd.read_sql(query_str, conn)

    tabla1 = ''
    for i in range(len(data1)):
        tabla1 += '<tr><td>' + \
            str(data1.iloc[i]['razon_social']) + '</td><td>' + \
            '${:,.2f}'.format(data1.iloc[i]['venta']) + '</td></tr>'
    
    """--------------------GRAFICA TOP CLIENTES EN TOTAL--------------------"""

    x = np.array(data1[['razon_social', 'venta']])
    fig3 = plt.figure(figsize=(10, 6))
    plt.barh(x[:, 0], x[:, 1], color=['#51FF00', '#1BFF00', '#00FF36', '#00FF87',
             '#00FFCD', '#00F7FF', '#00E4FF', '#00BDFF', '#0087FF', '#0068FF'])
    plt.xlabel('Valor ventas')
    plt.ylabel('Clientes')
    plt.title('Ventas x Mes 2021')

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    img_ventas2 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    """--------------------TABLA TOP PRODUCTOS EN TOTAL--------------------"""

    query_str = """
        SELECT TOP 10 prod.nombre as producto, COUNT(*) as cantidad FROM ventas_productoventa ven
        INNER JOIN productos_producto prod ON prod.id = ven.producto_id
        GROUP BY prod.nombre
        ORDER BY 2 DESC
    """

    data = pd.read_sql(query_str, conn)

    tabla2 = ''
    for i in range(len(data)):
        tabla2 += '<tr><td>' + \
            str(data.iloc[i]['producto']) + '</td><td>' + \
            str(data.iloc[i]['cantidad']) + '</td></tr>'

    """--------------------TABLA TOP CLIENTES TOTAL EN COMPRAS--------------------"""

    query_str = """
        SELECT TOP 10 cli.razon_social, COUNT(*) as total FROM ventas_venta ven
        INNER JOIN ventas_cliente cli ON cli.id = ven.cliente_id
        WHERE ven.cliente_id != 9883
        GROUP BY cli.razon_social
        ORDER BY 2 DESC
    """

    data = pd.read_sql(query_str, conn)

    tabla3 = ''
    for i in range(len(data)):
        tabla3 += '<tr><td>' + \
            str(data.iloc[i]['razon_social']) + '</td><td>' + \
            str(data.iloc[i]['total']) + '</td></tr>'

    m = {
        "before": img_ventas,
        "before2": img_ventas2,
        "tabla1": tabla,
        "tabla2": tabla1,
        "tabla3": tabla2,
        "tabla4": tabla3,
    }

    return render(request, 'index.html', context=m)


def clusters(request):

    """--------------------GRAFICA PRODUCTOS X CANTIDAD--------------------"""

    query_str = """
        SELECT MONTH(fecha) as mes, producto_id, SUM(cantidad) cantidad
        FROM   ventas_venta AS ven
        INNER JOIN ventas_productoventa AS v ON v.venta_id = ven.id
        GROUP BY month(fecha), producto_id
        ORDER BY 3 DESC
    """

    data = pd.read_sql(query_str, conn)

    dataset = data.drop(data[(data.cantidad > 300) |
                        (data.cantidad < 10)].index)
    X = np.array(dataset[['producto_id', 'cantidad']])
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('Producto ID')
    plt.ylabel('Cantidad')
    plt.title('Cantidad x producto')

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cluster1 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    """--------------------KMEANS PRODUCTOS POR CANTIDAD--------------------"""

    kmeans = KMeans(n_clusters=4).fit(X)
    centroides = kmeans.cluster_centers_
    etiquetas = kmeans.labels_
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap='rainbow')
    plt.scatter(centroides[:, 0], centroides[:, 1], color='black',
                marker='*', s=100)
    plt.xlabel('Producto')
    plt.ylabel('Cantidad')
    plt.title('KMeans Productos')

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cluster2 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    """--------------------GMM PRODUCTOS X CANTIDAD--------------------"""

    dataset = data.drop(data[(data.cantidad > 300) |
                        (data.cantidad < 10)].index)
    dataset=dataset.drop(["mes"], axis=1)
    print(dataset)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(dataset)
    pred = kmeans.predict(dataset)
    frame = pd.DataFrame(dataset)
    frame['cluster'] = pred
    frame.columns = ['producto_id', 'cantidad', 'cluster']
    #plotting results
    fig6 = plt.figure(figsize=(10,6))
    color=['blue','yellow','cyan', 'red']
    for k in range(0,4):
        dataset = frame[frame["cluster"]==k]
        plt.scatter(dataset["producto_id"],dataset["cantidad"],c=color[k])
    plt.xlabel('Productos')
    plt.ylabel('Cantidad')
    plt.title('GMM Clustering')

    tmpfile = BytesIO()
    fig6.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cluster6 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    """--------------------DBSCAN FAILED--------------------"""

    cluster = DBSCAN(eps=3, min_samples=3).fit(data)
    DBSCAN_dataset= data.copy()
    DBSCAN_dataset.loc[:, 'Cluster'] = cluster.labels_
    outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

    fig2, (axes) = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot('producto_id', 'cantidad',
                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                    ax=axes[0], palette='Set2', s=200)

    sns.scatterplot('mes', 'cantidad',
                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
       
                    ax=axes[1], palette='Set2', s=200)
    axes[0].scatter(outliers['producto_id'], outliers['cantidad'],
                    s=10, label='outliers', c="k")

    axes[1].scatter(outliers['mes'], outliers['cantidad'],
                    s=10, label='outliers', c="k")
    axes[0].legend()
    axes[1].legend()
    plt.setp(axes[0].get_legend().get_texts(), fontsize='12')
    plt.setp(axes[1].get_legend().get_texts(), fontsize='12')

    tmpfile = BytesIO()
    fig2.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cluster3 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    """--------------------DBSCAN--------------------"""

    fig3, (axes) = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot('producto_id', 'cantidad',
                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                    ax=axes[0], palette='Set2', s=200)

    sns.scatterplot('mes', 'cantidad',
                    data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                    ax=axes[1], palette='Set2', s=200)

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cluster4 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    

    query_str = """
        SELECT FORMAT(fecha, 'yyyyMMdd') fecha, sum(valor_total) valor FROM ventas_venta
        GROUP BY FORMAT(fecha, 'yyyyMMdd')
    """

    data1 = pd.read_sql(query_str, conn)

    cluster = DBSCAN(eps=12.5, min_samples=4).fit(data1)
    DBSCAN_dataset = data1.copy()
    DBSCAN_dataset.loc[:, 'Cluster'] = cluster.labels_

    fig4, (axes) = plt.subplots(1, figsize=(10, 5))
    sns.scatterplot('fecha', 'valor',
                    data=DBSCAN_dataset,
                    ax=axes, palette='Set2', s=200)

    axes.legend()
    plt.setp(axes.get_legend().get_texts(), fontsize='12')

    tmpfile = BytesIO()
    fig4.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cluster5 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    

    m = {
        "cluster1": cluster1,
        "cluster2": cluster2,
        "cluster3": cluster3,
        "cluster4": cluster4,
        "cluster5": cluster5,
        "cluster6": cluster6,
    }

    return render(request, 'cluster.html', context=m)

def prediccion(request):
    
    query_str = """
    SELECT MONTH(fecha) as mes, producto_id, SUM(cantidad) cantidad
    FROM   ventas_venta AS ven
    INNER JOIN ventas_productoventa AS v ON v.venta_id = ven.id
    GROUP BY month(fecha), producto_id
    ORDER BY 3 DESC"""

    data = pd.read_sql(query_str, conn)


    """
    Kmeans - grafica corriente
    """
    dataset = data.drop(data[(data.cantidad > 300) |
                        (data.cantidad < 10)].index)
    X = np.array(dataset[['producto_id', 'cantidad']])
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('Producto ID')
    plt.ylabel('Cantidad')
    plt.title('Cantidad x producto')

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    cluster1 = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    

    m = {
        "before": cluster1,
    }

    return render(request, 'prediccion.html', context=m)