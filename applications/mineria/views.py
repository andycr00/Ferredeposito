# Django
from django.shortcuts import render

# Lib
import pyodbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

    X = np.array(data[["mes", "total_mes"]])
    fig = plt.figure(figsize=(10, 6))
    plt.plot(X[:, 0], X[:, 1])
    plt.xlabel("Valor ventas")
    plt.ylabel("mes")
    plt.title("Ventas x Mes 2021")

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    img_ventas = "<img src='data:image/png;base64,{}'>".format(encoded)

    """--------------------TABLA VENTAS POR MES--------------------"""

    meses = [
        "Enero",
        "Febrero",
        "Marzo",
        "Abril",
        "Mayo",
        "Junio",
        "Julio",
        "Agosto",
        "Septiembre",
        "Octubre",
        "Noviembre",
        "Diciembre",
    ]

    tabla = ""

    for i in range(len(data)):
        tabla += (
            "<tr><td>"
            + meses[i]
            + "</td><td>"
            + "${:,.2f}".format(data.iloc[i]["total_mes"])
            + "</td></tr>"
        )

    """--------------------TABLA TOP CLIENTES EN TOTAL--------------------"""

    query_str = """
        SELECT TOP 10 cliente_id, razon_social, SUM(valor_total) as venta FROM ventas_venta ven
        INNER JOIN ventas_cliente cli ON cli.id = ven.cliente_id
        WHERE razon_social != ' PUBLICO GENERAL '
        GROUP BY cliente_id, razon_social
        ORDER BY 3 DESC
    """

    data1 = pd.read_sql(query_str, conn)

    tabla1 = ""
    for i in range(len(data1)):
        tabla1 += (
            "<tr><td>"
            + str(data1.iloc[i]["razon_social"])
            + "</td><td>"
            + "${:,.2f}".format(data1.iloc[i]["venta"])
            + "</td></tr>"
        )

    """--------------------GRAFICA TOP CLIENTES EN TOTAL--------------------"""

    x = np.array(data1[["razon_social", "venta"]])
    fig3 = plt.figure(figsize=(10, 6))
    plt.barh(
        x[:, 0],
        x[:, 1],
        color=[
            "#51FF00",
            "#1BFF00",
            "#00FF36",
            "#00FF87",
            "#00FFCD",
            "#00F7FF",
            "#00E4FF",
            "#00BDFF",
            "#0087FF",
            "#0068FF",
        ],
    )
    plt.xlabel("Valor ventas")
    plt.ylabel("Clientes")
    plt.title("Ventas x Mes 2021")

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    img_ventas2 = "<img src='data:image/png;base64,{}'>".format(encoded)

    """--------------------TABLA TOP PRODUCTOS EN TOTAL--------------------"""

    query_str = """
        SELECT TOP 10 prod.nombre as producto, COUNT(*) as cantidad FROM ventas_productoventa ven
        INNER JOIN productos_producto prod ON prod.id = ven.producto_id
        GROUP BY prod.nombre
        ORDER BY 2 DESC
    """

    data = pd.read_sql(query_str, conn)

    tabla2 = ""
    for i in range(len(data)):
        tabla2 += (
            "<tr><td>"
            + str(data.iloc[i]["producto"])
            + "</td><td>"
            + str(data.iloc[i]["cantidad"])
            + "</td></tr>"
        )

    """--------------------TABLA TOP CLIENTES TOTAL EN COMPRAS--------------------"""

    query_str = """
        SELECT TOP 10 cli.razon_social, COUNT(*) as total FROM ventas_venta ven
        INNER JOIN ventas_cliente cli ON cli.id = ven.cliente_id
        WHERE ven.cliente_id != 9883
        GROUP BY cli.razon_social
        ORDER BY 2 DESC
    """

    data = pd.read_sql(query_str, conn)

    tabla3 = ""
    for i in range(len(data)):
        tabla3 += (
            "<tr><td>"
            + str(data.iloc[i]["razon_social"])
            + "</td><td>"
            + str(data.iloc[i]["total"])
            + "</td></tr>"
        )

    m = {
        "before": img_ventas,
        "before2": img_ventas2,
        "tabla1": tabla,
        "tabla2": tabla1,
        "tabla3": tabla2,
        "tabla4": tabla3,
    }

    return render(request, "index.html", context=m)


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
    X = np.array(dataset[["producto_id", "cantidad"]])
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("Producto ID")
    plt.ylabel("Cantidad")
    plt.title("Cantidad x producto")

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cluster1 = "<img src='data:image/png;base64,{}'>".format(encoded)

    """--------------------KMEANS PRODUCTOS POR CANTIDAD--------------------"""

    kmeans = KMeans(n_clusters=4).fit(X)
    centroides = kmeans.cluster_centers_
    etiquetas = kmeans.labels_
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap="rainbow")
    plt.scatter(centroides[:, 0], centroides[:, 1],
                color="black", marker="*", s=100)
    plt.xlabel("Producto")
    plt.ylabel("Cantidad")
    plt.title("KMeans Productos")

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cluster2 = "<img src='data:image/png;base64,{}'>".format(encoded)

    """--------------------GMM PRODUCTOS X CANTIDAD--------------------"""

    dataset = data.drop(data[(data.cantidad > 300) |
                        (data.cantidad < 10)].index)
    dataset = dataset.drop(["mes"], axis=1)
    print(dataset)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(dataset)
    pred = kmeans.predict(dataset)
    frame = pd.DataFrame(dataset)
    frame["cluster"] = pred
    frame.columns = ["producto_id", "cantidad", "cluster"]
    # plotting results
    fig6 = plt.figure(figsize=(10, 6))
    color = ["blue", "yellow", "cyan", "red"]
    for k in range(0, 4):
        dataset = frame[frame["cluster"] == k]
        plt.scatter(dataset["producto_id"], dataset["cantidad"], c=color[k])
    plt.xlabel("Productos")
    plt.ylabel("Cantidad")
    plt.title("GMM Clustering")

    tmpfile = BytesIO()
    fig6.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cluster6 = "<img src='data:image/png;base64,{}'>".format(encoded)

    """--------------------DBSCAN FAILED--------------------"""

    cluster = DBSCAN(eps=3, min_samples=3).fit(data)
    DBSCAN_dataset = data.copy()
    DBSCAN_dataset.loc[:, "Cluster"] = cluster.labels_
    outliers = DBSCAN_dataset[DBSCAN_dataset["Cluster"] == -1]

    fig2, (axes) = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(
        "producto_id",
        "cantidad",
        data=DBSCAN_dataset[DBSCAN_dataset["Cluster"] != -1],
        ax=axes[0],
        palette="Set2",
        s=200,
    )

    sns.scatterplot(
        "mes",
        "cantidad",
        data=DBSCAN_dataset[DBSCAN_dataset["Cluster"] != -1],
        ax=axes[1],
        palette="Set2",
        s=200,
    )
    axes[0].scatter(
        outliers["producto_id"], outliers["cantidad"], s=10, label="outliers", c="k"
    )

    axes[1].scatter(
        outliers["mes"], outliers["cantidad"], s=10, label="outliers", c="k"
    )
    axes[0].legend()
    axes[1].legend()
    plt.setp(axes[0].get_legend().get_texts(), fontsize="12")
    plt.setp(axes[1].get_legend().get_texts(), fontsize="12")

    tmpfile = BytesIO()
    fig2.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cluster3 = "<img src='data:image/png;base64,{}'>".format(encoded)

    """--------------------DBSCAN FAILED--------------------"""

    fig3, (axes) = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(
        "producto_id",
        "cantidad",
        data=DBSCAN_dataset[DBSCAN_dataset["Cluster"] != -1],
        ax=axes[0],
        palette="Set2",
        s=200,
    )

    sns.scatterplot(
        "mes",
        "cantidad",
        data=DBSCAN_dataset[DBSCAN_dataset["Cluster"] != -1],
        ax=axes[1],
        palette="Set2",
        s=200,
    )

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cluster4 = "<img src='data:image/png;base64,{}'>".format(encoded)

    """--------------------DBSCAN FAILED--------------------"""

    query_str = """
        SELECT FORMAT(fecha, 'yyyyMMdd') fecha, sum(valor_total) valor FROM ventas_venta
        GROUP BY FORMAT(fecha, 'yyyyMMdd')
    """

    data1 = pd.read_sql(query_str, conn)

    cluster = DBSCAN(eps=12.5, min_samples=4).fit(data1)
    DBSCAN_dataset = data1.copy()
    DBSCAN_dataset.loc[:, "Cluster"] = cluster.labels_

    fig4, (axes) = plt.subplots(1, figsize=(10, 5))
    sns.scatterplot(
        "fecha", "valor", data=DBSCAN_dataset, ax=axes, palette="Set2", s=200
    )

    axes.legend()
    plt.setp(axes.get_legend().get_texts(), fontsize="12")

    tmpfile = BytesIO()
    fig4.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cluster5 = "<img src='data:image/png;base64,{}'>".format(encoded)

    m = {
        "cluster1": cluster1,
        "cluster2": cluster2,
        "cluster3": cluster3,
        "cluster4": cluster4,
        "cluster5": cluster5,
        "cluster6": cluster6,
    }

    return render(request, "cluster.html", context=m)


def prediccion(request):
    """--------------------LINEAR REGRESSION FAILED--------------------"""

    query_str = """
        SELECT MONTH(fecha) mes, day(fecha) dia , DATEPART(HOUR, fecha) 
        hora, categoria_id from ventas_venta vent
        INNER JOIN ventas_productoventa ven ON vent.id = ven.venta_id
        INNER JOIN productos_producto prod ON prod.id = ven.producto_id
        WHERE categoria_id IN (2,13,7,6,8,15,19,5,3,23)
    """

    data = pd.read_sql(query_str, conn)

    Lista_condicion = [
        (data["categoria_id"] == 2),
        (data["categoria_id"] == 13),
        (data["categoria_id"] == 7),
        (data["categoria_id"] == 6),
        (data["categoria_id"] == 8),
        (data["categoria_id"] == 15),
        (data["categoria_id"] == 19),
        (data["categoria_id"] == 5),
        (data["categoria_id"] == 3),
        (data["categoria_id"] == 23)
    ]
    Lista_clasificacion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    data["que_tanto"] = np.select(
        Lista_condicion, Lista_clasificacion, default="no especificado")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    regressor.predict(X_test)
    RegresionLineal = (regressor.score(X_train, y_train) * 100).round(3)

    Lista_condicion = [
        (data["categoria_id"] == 2),
        (data["categoria_id"] == 13),
        (data["categoria_id"] == 7),
        (data["categoria_id"] == 6),
        (data["categoria_id"] == 8),
        (data["categoria_id"] == 15),
        (data["categoria_id"] == 19),
        (data["categoria_id"] == 5),
        (data["categoria_id"] == 3),
        (data["categoria_id"] == 23)
    ]
    Lista_clasificacion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    data["que_tanto"] = np.select(
        Lista_condicion, Lista_clasificacion, default=10)

    x = data.iloc[:, 1]
    y = data.iloc[:, -1]

    myline = np.linspace(1, 31, 100, dtype=int)
    mymodel = np.poly1d(np.polyfit(x, y.astype('f'), 3))
    plt.scatter(x, y.astype('f'))
    plt.plot(myline, mymodel(myline))

    speed = (r2_score(y, mymodel(x)) * 100).round(3)

    Lista_condicion = [
        (data["hora"] < 12),
        (data["hora"] >= 12),
    ]

    Lista_clasificacion = [0, 1]

    data["que_tanto"] = np.select(
        Lista_condicion, Lista_clasificacion, default=10)

    y = data['que_tanto']
    X = data.iloc[:, :-1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    classifier = LogisticRegression(random_state=0, max_iter=300)
    classifier = classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision = (precision_score(Y_test, Y_pred)*100).round(3)

    obj = {
        "Regresion Lineal": RegresionLineal,
        "Regresion Polinomial": speed,
        "Regresion Logística": precision,
    }

    tabla = ""

    for val in obj:
        tabla += (
            "<tr><td>"
            + val
            + "</td><td>"
            + str(obj[val]) + ' %'
            + "</td></tr>"
        )

    names = []
    values = []
    for i in obj:
        names.append(i)
        values.append(obj[i])
    fig3 = plt.figure(figsize=(10, 6))
    plt.bar(
        names,
        values,
        color=[
            "#45B8AC",
            "#EFC050",
            "#5B5EA6",
            "#55B4B0",
            "#98B4D4",
            "#FF6F61",
        ],
    )
    plt.ylabel("Precision")
    plt.title("Precision en la prediccion")

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    predTable = "<img src='data:image/png;base64,{}'>".format(encoded)

    m = {
        "tabla_precision": tabla,
        "grafica": predTable
    }

    return render(request, "prediccion.html", context=m)


def clasificacion(request):
    """--------------------CLASIFICACION KMEANS--------------------"""

    query_str = """
        SELECT MONTH(fecha) as mes, producto_id, SUM(cantidad) cantidad
        FROM   ventas_venta AS ven	
        INNER JOIN ventas_productoventa AS v ON v.venta_id = ven.id
        GROUP BY month(fecha), producto_id
        ORDER BY 3 DESC
    """

    data = pd.read_sql(query_str, conn)

    Lista_condicion = [
        (data["cantidad"] <50),
        (data["cantidad"] >= 50),
    ]

    Lista_clasificacion = [0, 1]

    data["que_tanto"] = np.select(
        Lista_condicion, Lista_clasificacion, default=10)

    y = data['que_tanto']
    X = data.iloc[:, :-1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    classifier = LogisticRegression(random_state=0, max_iter=300)
    classifier = classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision1 = (precision_score(Y_test, Y_pred)*100).round(3)

    classifier.fit(X_train, Y_train)
    fig1, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig1.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm1 = "<img src='data:image/png;base64,{}'>".format(encoded)    

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision2 = (precision_score(Y_test, Y_pred) * 100).round(3)
    fig2, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig2.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm2 = "<img src='data:image/png;base64,{}'>".format(encoded)  

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision3 = (precision_score(Y_test, Y_pred) * 100).round(3)
    fig3, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm3 = "<img src='data:image/png;base64,{}'>".format(encoded)  

    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision4 = (precision_score(Y_test, Y_pred) * 100).round(3)
    fig4, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig4.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm4 = "<img src='data:image/png;base64,{}'>".format(encoded)  

    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    precision5 = (precision_score(Y_test, Y_pred)*100).round(3)

    fig5, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig5.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm5 = "<img src='data:image/png;base64,{}'>".format(encoded)  


    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision6 = (precision_score(Y_test, Y_pred) * 100).round(3)

    fig6, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig6.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm6 = "<img src='data:image/png;base64,{}'>".format(encoded)  


    obj = {
        "Regresion Logistica": precision1,
        "KNeighbors Classifier": precision2,
        "Vectores Soporte": precision3,
        "Kernel SVM": precision4,
        "Naïve Bayes": precision5,
        "Arbol de decision": precision6
    }

    tabla = ""

    for val in obj:
        tabla += (
            "<tr><td>"
            + val
            + "</td><td>"
            + str(obj[val]) + ' %'
            + "</td></tr>"
        )
    names = []
    values = []
    for i in obj:
        names.append(i)
        values.append(obj[i])
    fig3 = plt.figure(figsize=(10, 6))
    plt.bar(
        names,
        values,
        color=[
            "#45B8AC",
            "#EFC050",
            "#5B5EA6",
            "#55B4B0",
            "#98B4D4",
            "#FF6F61",
        ],
    )
    plt.ylabel("Precision")
    plt.title("Precision en la prediccion")

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    predTable = "<img src='data:image/png;base64,{}'>".format(encoded)

    m = {
        "tabla_precision": tabla,
        "grafica": predTable,
        "cm1": cm1,
        "cm2": cm2,
        "cm3": cm3,
        "cm4": cm4,
        "cm5": cm5,
        "cm6": cm6,
    }

    return render(request, "clasificacion.html", context=m)

def clasificacion2(request):
    """--------------------LINEAR REGRESSION FAILED--------------------"""

    query_str = """
        SELECT MONTH(fecha) mes, day(fecha) dia , DATEPART(HOUR, fecha) 
        hora, categoria_id from ventas_venta vent
        INNER JOIN ventas_productoventa ven ON vent.id = ven.venta_id
        INNER JOIN productos_producto prod ON prod.id = ven.producto_id
        WHERE categoria_id IN (2,13,7,6,8,15,19,5,3,23)
    """

    data = pd.read_sql(query_str, conn)

    Lista_condicion = [
        (data["hora"] < 12),
        (data["hora"] >= 12),
    ]

    Lista_clasificacion = [0, 1]

    data["que_tanto"] = np.select(
        Lista_condicion, Lista_clasificacion, default=10)

    y = data['que_tanto']
    X = data.iloc[:, :-1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    classifier = LogisticRegression(random_state=0, max_iter=300)
    classifier = classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision1 = (precision_score(Y_test, Y_pred)*100).round(3)

    classifier.fit(X_train, Y_train)
    fig1, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig1.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm1 = "<img src='data:image/png;base64,{}'>".format(encoded)    

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision2 = (precision_score(Y_test, Y_pred) * 100).round(3)
    fig2, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig2.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm2 = "<img src='data:image/png;base64,{}'>".format(encoded)  

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision3 = (precision_score(Y_test, Y_pred) * 100).round(3)
    fig3, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm3 = "<img src='data:image/png;base64,{}'>".format(encoded)  

    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision4 = (precision_score(Y_test, Y_pred) * 100).round(3)
    fig4, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig4.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm4 = "<img src='data:image/png;base64,{}'>".format(encoded)  

    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    precision5 = (precision_score(Y_test, Y_pred)*100).round(3)

    fig5, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig5.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm5 = "<img src='data:image/png;base64,{}'>".format(encoded)  


    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    precision6 = (precision_score(Y_test, Y_pred) * 100).round(3)

    fig6, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(classifier, X_test, Y_test, ax=ax)

    tmpfile = BytesIO()
    fig6.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    cm6 = "<img src='data:image/png;base64,{}'>".format(encoded)  


    obj = {
        "R. Logistica": precision1,
        "KNeighbors": precision2,
        "Vect. Soporte": precision3,
        "Kernel SVM": precision4,
        "Naïve Bayes": precision5,
        "Arbol decision": precision6
    }

    tabla = ""

    for val in obj:
        tabla += (
            "<tr><td>"
            + val
            + "</td><td>"
            + str(obj[val]) + ' %'
            + "</td></tr>"
        )
    names = []
    values = []
    for i in obj:
        names.append(i)
        values.append(obj[i])
    fig3 = plt.figure(figsize=(10, 6))
    plt.bar(
        names,
        values,
        color=[
            "#45B8AC",
            "#EFC050",
            "#5B5EA6",
            "#55B4B0",
            "#98B4D4",
            "#FF6F61",
        ],
    )
    plt.ylabel("Precision")
    plt.title("Precision en la prediccion")

    tmpfile = BytesIO()
    fig3.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    predTable = "<img src='data:image/png;base64,{}'>".format(encoded)

    m = {
        "tabla_precision": tabla,
        "grafica": predTable,
        "cm1": cm1,
        "cm2": cm2,
        "cm3": cm3,
        "cm4": cm4,
        "cm5": cm5,
        "cm6": cm6,
    }

    return render(request, "clasificacion.html", context=m)
