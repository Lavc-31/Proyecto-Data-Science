# coding=utf-8
import cv2
import numpy as np
from flask import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

"""
# Se importa la libreria "numpy" la cual nos brinda una amplia gama de funcionalidades matematicas.
# Se importa la libreria "tensorflow" la cual brinda funcionalidades de preprocesamiento y permite definir las capas del modelo.
# Se importa la libreria "cv2" correspondiente a OpenCV la cual nos da acceso a multiples funcionalidades de manejo y tratamiento de imagenes.
# Se importa la libreria "flask" la cual nos permite montar APIs y manejar sus rutas de forma sencilla.
"""

app = Flask(__name__)
@app.route('/', methods=['POST','GET'])

# Se define la variable app, la cual guarda una instancia de la clase Flask permitiendo la conexion http.
# Se indica una ruta de la URL y los metodos aceptados para dicha ruta.


def clasificador():
	reconstructed_model = keras.models.load_model("modelo-3era-edad")
	print("modelo cargado!!!!")

	if (request.method == "POST"):
		image = request.files.get('img')
		image_name = request.files['img'].filename
		image_bytes = image.read()
		print("nombre es: ",image_name)

		if (image_name != ''):
			foto = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.byte), flags=cv2.IMREAD_COLOR)
			foto_final = tf.keras.preprocessing.image.img_to_array(foto)
			foto_final = np.expand_dims(foto_final, axis=0)
			foto_final = np.resize(foto_final,(1,96,96,3))

			foto_tensor = tf.convert_to_tensor(foto_final, dtype=float, name='tensor1')
			#print("tensor is: ", foto_tensor)
			preds = reconstructed_model.predict_on_batch(foto_tensor)
			#print("preds es: ", preds)

			tags = ["-30", "31-35", "36-40", "41-50", "51-60", "61+"]

			result = tags[np.argmax(preds)]
			print("result es: ", result)
			#preds_as_list = preds.tolist()

			return jsonify(resultado=result)
			#return jsonify(resultado=result, probabilidades=preds_as_list)

		return render_template('recibir-foto.html')

	return render_template('recibir-foto.html')

"""
# Funcion clasificador()
# Input: None (Esta constantemente a la escucha de solicitudes GET o POST provenientes del sitio web)
# Output: (Un json con los resultados de las probabilidades y la clasificacion realizada o en caso de no existir esa informacion se muestra el sitio web)
# Funcionamiento: Se lee una imagen a partir de una solicitud POST, se preprocesa y se entrega como parametro a la funcion de prediccion de tensorflow,
haciendo uso de los modelos reconstruidos.
"""

if __name__ == '__main__':
	app.run(debug=True)

"""
# Se ejecuta la instancia de la clase Flask previamente definida.
"""