import base64
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import exposure, color
import streamlit as st


def save_temp_image(image_to_save):
    """
    Guarda una imagen en un archivo temporal.

    Args:
        image_to_save (numpy.ndarray): La imagen que se va a guardar.

    Returns:
        Tuple[str, str]: Una tupla que contiene la ruta del \
        archivo de imagen temporal y el directorio temporal.
    """

    # Se crea un directorio temporal para almacenar el archivo de imagen.
    temp_directory = tempfile.mkdtemp()
    temp_image_path = temp_directory + "/imagen_recortada.png"

    # Se guarda la imagen en el archivo temporal.
    io.imsave(temp_image_path, image_to_save)
    return temp_image_path, temp_directory


def image_to_base64(image_to_convert):
    """
    Convierte una imagen en formato base64.

    Args:
        image_to_convert (numpy.ndarray): La imagen que se va a convertir.

    Returns:
        str: La representación en base64 de la imagen.
    """

    # Se muestra la imagen en un búfer.
    image_buffer = io.imshow(image_to_convert, plugin='matplotlib')

    # Se dibuja el búfer para obtener la representación de la imagen.
    image_buffer.figure.canvas.draw()

    # Se codifica la imagen en base64 y se devuelve como una cadena.
    return base64.b64encode(image_buffer.figure.canvas.tostring_rgb()).decode()


# Se pinta el título de la aplicación.
st.title("Procesamiento de imágenes")

# Se pinta el input para cargar la imagen que se\
# desea analizar.
uploaded_image = st.file_uploader("Cargar una imagen", type=["jpg", "png", "jpeg"])

# Se define la información que se debe mostrar\
# después de cargar la imagen.
if uploaded_image is not None:

    # Se pinta un slider que permite seleccionar el\
    # nivel de contraste de la imagen cargada.
    contrast_level = st.slider("Nivel de Contraste", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Se convierte la imagen cargada con streamlit a una\
    # imagen de scikit-image
    image = io.imread(uploaded_image)

    # Si la imagen es RGBA la convierte a RGB
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)

    # Se definen las columnas en dónde se compara la imagen original\
    # con la imagen y el contraste definido.
    col1, col2 = st.columns(2)
    with col1:
        # Se pinta la imagen original.
        st.image(image, caption="Imagen Original", use_column_width=True)

        # Se definen los maximos de los sliders para recortar la imagen
        max_x, max_y = image.shape[1], image.shape[0]

        # Se definen los sliders para recortar la imagen
        crop_x = st.sidebar.slider("Recorte X (inicio) (píxeles)", 0, max_x, 0)
        crop_y = st.sidebar.slider("Recorte Y (inicio) (píxeles)", 0, max_y, 0)
        crop_width = st.sidebar.slider("Ancho del Recorte (píxeles)", 0, max_x, max_x)
        crop_height = st.sidebar.slider("Alto del Recorte (píxeles)", 0, max_y, max_y)

        # Se recorta la imagen como un arreglo de numpy
        cropped_image = image[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

        # Se pinta la imagen recortada
        st.image(cropped_image, caption="Imagen Recortada", use_column_width=True)
    with col2:
        # Se ajusta el contraste de la imagen con los datos del slider.
        contrast_image = exposure.adjust_gamma(image, contrast_level)

        # Se pinta la imagen alterada.
        st.image(contrast_image, caption="Ajuste de Contraste", use_column_width=True)

        # Se genera un histograma del contraste de la imagen.
        equalized_image = exposure.equalize_hist(contrast_image)
        equalized_hist, _ = np.histogram(equalized_image, bins=256, range=(0, 1))

        # Se configura la imagen con matplotlib
        fig, ax = plt.subplots()
        ax.plot(equalized_hist, color='black')
        ax.set_title("Histograma del Contraste")
        ax.set_xlabel("Valor de Píxel")
        ax.set_ylabel("Frecuencia")

        # Se pinta el histograma
        st.pyplot(fig)


    # Se define el botón para descargar la imagen recortada
    if st.sidebar.button("Descargar Imagen Recortada"):
        # Se guarda temporalmente la imagen
        path, directory = save_temp_image(cropped_image)
        with open(path, "rb") as file:
            image_bytes = file.read()
            # Se genera el anchor para descargar la imagen
            st.sidebar.markdown(
                f'<a href="data:image/png;base64,{base64.b64encode(image_bytes).decode()}" '
                f'download="imagen_recortada.png">Haga clic aquí para descargar la imagen recortada</a>',
                unsafe_allow_html=True
            )
        # Se limpia el director temporal
        shutil.rmtree(directory)
