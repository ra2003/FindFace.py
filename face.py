import face_recognition
from PIL import Image, ImageDraw

img_of_snow = face_recognition.load_image_file('./known/johnsnow.jpg')
snow_encoding = face_recognition.face_encodings(img_of_snow)[0]

img_of_arya = face_recognition.load_image_file('./known/arya stark.jpg')
arya_encoding = face_recognition.face_encodings(img_of_arya)[0]

img_of_hound = face_recognition.load_image_file('./known/sandor clegane.jpg')
hound_encoding = face_recognition.face_encodings(img_of_hound)[0]

img_of_jaime = face_recognition.load_image_file('./known/jaime.jpg')
jaime_encoding = face_recognition.face_encodings(img_of_jaime)[0]

img_of_oberyn = face_recognition.load_image_file('./known/oberyn.jpg')
oberyn_encoding = face_recognition.face_encodings(img_of_oberyn)[0]

img_of_ygritte = face_recognition.load_image_file('./known/ygritte.jpg')
ygritte_encoding = face_recognition.face_encodings(img_of_ygritte)[0]

img_of_sam = face_recognition.load_image_file('./known/samwell.jpg')
sam_encoding = face_recognition.face_encodings(img_of_sam)[0]

img_of_sansa = face_recognition.load_image_file('./known/sansa.jpg')
sansa_encoding = face_recognition.face_encodings(img_of_sansa)[0]

img_of_brienne = face_recognition.load_image_file('./known/brienne.jpg')
brienne_encoding = face_recognition.face_encodings(img_of_brienne)[0]

img_of_margaery = face_recognition.load_image_file('./known/margaery.jpg')
margaery_encoding = face_recognition.face_encodings(img_of_margaery)[0]

print('Files Loaded')

known_face_encodings = [
    snow_encoding,
    sam_encoding,
    arya_encoding,
    margaery_encoding,
    oberyn_encoding,
    hound_encoding,
    brienne_encoding,
    ygritte_encoding,
    jaime_encoding,
    sansa_encoding
]

known_face_names = [
    "She is my queen",
    "zero kill winner",
    "Arya the explorer",
    "Literally roasted",
    "Crushed skull guy",
    "I hate toys",
    "One knight stand",
    "you know nothing, JS",
    "Kingslayer",
    "Queen in the north"
]

print('Names loaded')

img_to_detect = face_recognition.load_image_file('./unknown/group.jpg')

face_locations = face_recognition.face_locations(img_to_detect)
face_encodings = face_recognition.face_encodings(img_to_detect, face_locations)

#PIL Conversion

pil_image = Image.fromarray(img_to_detect)

#Draw instance

draw = ImageDraw.Draw(pil_image)

print('Loop entered')

#LOOPS

for(top, right, bottom, left), face_encoding in zip(face_locations,face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = " Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    
    #DRAW
    draw.rectangle(((left,top),(right,bottom)), outline= (0,0,0))

    text_width , text_height = draw.textsize(name)
    draw.rectangle(((left,bottom - text_height - 5), (right + 30,bottom)), fill=(0,0,0), outline=(0,0,0))
    draw.text((left + 5, bottom - text_height - 5), name, fill=(255,255,255,255))

del draw
pil_image.show()
pil_image.save('detected.jpg')
#Ignore this line, I just wrote this line just to make loc as 100! :D