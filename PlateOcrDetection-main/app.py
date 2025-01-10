from flask import Flask, jsonify, request, url_for, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
from loguru import logger
import easyocr
import cv2
import re

# Configurações principais
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = r'C:\Users\CASA\Downloads\PlateOcrDetection-main\uploads'
OUTPUT_FOLDER = r'C:\Users\CASA\Downloads\PlateOcrDetection-main\outputs'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Função para verificar se a extensão do arquivo é permitida


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Função para desenhar as caixas delimitadoras na imagem


def draw_boxes(image_path, results):
    image = cv2.imread(image_path)
    for result in results:
        # Converta as coordenadas para inteiros
        top_left = tuple(map(int, result[0][0]))
        bottom_right = tuple(map(int, result[0][2]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        text = result[1]
        cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Cria a pasta de saída, se não existir
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Define o caminho de saída para a imagem
    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    return output_path

# Classe para análise de dados de placas


class PlateDataAnalysis:
    def __init__(self):
        self.reader = easyocr.Reader(
            ['pt', 'en'], gpu=False)  # GPU=True se disponível

    def read_text_from_image(self, image_path, decoder='beamsearch'):
        results = self.reader.readtext(image_path, decoder=decoder)
        return results

    def filter_plates(self, results):
        potential_plates = []
        pattern = r'^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$'
        for result in results:
            text, confidence = result[1], result[2]
            logger.info(f'Extracted text: {text} | Confidence: {confidence}')
            is_valid_plate = re.match(pattern, text) is not None
            if confidence > 0.3:
                potential_plates.append({
                    'text': text,
                    'confidence': confidence,
                    'valid': is_valid_plate
                })
        return potential_plates if potential_plates else None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f'Image saved at {file_path}')

        # Processar a imagem e realizar OCR
        plate_analysis = PlateDataAnalysis()
        texts = plate_analysis.read_text_from_image(file_path, 'beamsearch')
        text_plate = plate_analysis.filter_plates(texts)

        # Filtrar apenas placas válidas
        valid_plates = [
            plate for plate in text_plate if plate['valid']] if text_plate else None

        # Desenhar caixas nas letras detectadas
        output_image_path = draw_boxes(file_path, texts)
        logger.info(f'Processed image saved at {output_image_path}')
        if not os.path.exists(output_image_path):
            logger.error(
                f'Image was not saved correctly at {output_image_path}')

        # Preparar a resposta
        response = {
            'image_url': url_for('output_file', filename=os.path.basename(output_image_path), _external=True),
            'detected_texts': [{'text': item[1], 'confidence': item[2]} for item in texts],
            'plates': [{'plate': plate['text'], 'confidence': plate['confidence'], 'valid': plate['valid']}
                       for plate in valid_plates] if valid_plates else 'Nenhuma placa válida detectada.'
        }

        return jsonify(response), 200

    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    app.run(host='0.0.0.0', port=5000, debug=True)
