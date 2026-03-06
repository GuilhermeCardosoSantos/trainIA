from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="pt")

result = ocr.predict("image.jpg")

for line in result:
    for word in line:
        print(word[1][0], word[1][1])