
---

# API de Detecção de Placas (OCR)

Esta API permite o upload de imagens, realiza o Reconhecimento Óptico de Caracteres (OCR) para detectar textos (como placas de veículos) e retorna a imagem processada com caixas delimitadoras nas letras detectadas. A API também tenta filtrar e identificar possíveis números de placas de veículos.

## Funcionalidades
- Upload de uma imagem (JPEG/PNG)
- Detecção de texto usando EasyOCR
- Filtragem de possíveis números de placas de veículos
- Desenho de caixas delimitadoras no texto detectado na imagem
- Servir a imagem processada com as caixas desenhadas

## Endpoints

### 1. Fazer Upload de Imagem para OCR
**`POST /upload`**

Este endpoint aceita uma imagem, realiza OCR para detectar texto e retorna possíveis números de placas de veículos junto com a imagem processada que contém caixas delimitadoras em torno das letras detectadas.

#### Requisição
- **Método**: POST
- **Content-Type**: `multipart/form-data`
- **Parâmetros**:
  - `image`: Arquivo de imagem a ser enviado. Somente os formatos `png`, `jpg` e `jpeg` são suportados.

#### Resposta
- **Sucesso (200)**:
  ```json
  {
    "image_url": "http://<seu_endereço_servidor>/outputs/<nome_da_imagem_processada>",
    "detected_texts": [
      {"text": "XYZ1234", "confidence": 0.92},
      {"text": "ABC5678", "confidence": 0.85}
    ],
    "plates": [
      {"text": "XYZ1234", "confidence": 0.92}
    ]
  }
  ```
  - **`image_url`**: URL da imagem processada com caixas desenhadas.
  - **`detected_texts`**: Lista de todos os textos detectados na imagem, com níveis de confiança.
  - **`plates`**: Lista de possíveis números de placas de veículos detectados, com níveis de confiança.
  
- **Falha (400)**:
  ```json
  {
    "error": "Tipo de arquivo não permitido"
  }
  ```

### 2. Servir Arquivos Enviados
**`GET /uploads/<filename>`**

Este endpoint permite que você recupere a imagem original enviada.

- **Exemplo**: `GET /uploads/carro.jpeg`

### 3. Servir Imagem Processada com Caixas Delimitadoras
**`GET /outputs/<filename>`**

Este endpoint permite que você recupere a imagem processada com caixas desenhadas ao redor do texto detectado.

- **Exemplo**: `GET /outputs/carro_processado.jpeg`

## Instruções de Configuração

### 1. Clonar o Repositório

```bash
git clone https://github.com/seuusuario/ocr-plate-detection-api.git
cd ocr-plate-detection-api
```

### 2. Instalar Dependências

Certifique-se de ter o Python 3.x e o `pip` instalados. Em seguida, instale os pacotes necessários:

```bash
pip install -r requirements.txt
```

### 3. Executar a API

Execute a aplicação Flask:

```bash
python app.py
```

A API estará disponível em `http://127.0.0.1:5000` por padrão.

### 4. Fazer Upload de uma Imagem

Você pode testar a API enviando uma requisição POST com um arquivo de imagem:

```bash
curl -X POST -F 'image=@caminho_para_sua_imagem.jpg' http://127.0.0.1:5000/upload
```

### Requisitos
- Python 3.x
- Flask
- EasyOCR
- OpenCV
- Loguru

### Estrutura de Arquivos
```
.
├── app.py               # Arquivo principal da aplicação
├── uploads/             # Diretório para imagens enviadas
├── outputs/             # Diretório para imagens processadas com caixas delimitadoras
└── requirements.txt     # Lista de dependências
```

---
