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

img_of_cersie = face_recognition.load_image_file('./known/cersie.jpg')
cersie_encoding = face_recognition.face_encodings(img_of_cersie)[0]

img_of_theon = face_recognition.load_image_file('./known/theon.jpg')
theon_encoding = face_recognition.face_encodings(img_of_theon)[0]

img_of_sam = face_recognition.load_image_file('./known/samwell.jpg')
sam_encoding = face_recognition.face_encodings(img_of_sam)[0]

img_of_sansa = face_recognition.load_image_file('./known/sansa.jpg')
sansa_encoding = face_recognition.face_encodings(img_of_sansa)[0]

img_of_brienne = face_recognition.load_image_file('./known/brienne.jpg')
brienne_encoding = face_recognition.face_encodings(img_of_brienne)[0]

img_of_serdavos = face_recognition.load_image_file('./known/serdavos.jpg')
serdavos_encoding = face_recognition.face_encodings(img_of_serdavos)[0]

img_of_bran = face_recognition.load_image_file('./known/bran.jpg')
bran_encoding = face_recognition.face_encodings(img_of_bran)[0]

img_of_daenerys = face_recognition.load_image_file('./known/daenerys.jpg')
daenerys_encoding = face_recognition.face_encodings(img_of_daenerys)[0]

img_of_euron = face_recognition.load_image_file('./known/euron.jpg')
euron_encoding = face_recognition.face_encodings(img_of_euron)[0]

img_of_gilly = face_recognition.load_image_file('./known/gilly.jpg')
gilly_encoding = face_recognition.face_encodings(img_of_gilly)[0]

img_of_greyworm = face_recognition.load_image_file('./known/greyworm.jpg')
greyworm_encoding = face_recognition.face_encodings(img_of_greyworm)[0]

img_of_jorah = face_recognition.load_image_file('./known/jorah.jpg')
jorah_encoding = face_recognition.face_encodings(img_of_jorah)[0]

img_of_tyrion = face_recognition.load_image_file('./known/tyrion.jpg')
tyrion_encoding = face_recognition.face_encodings(img_of_tyrion)[0]

img_of_mellisandre = face_recognition.load_image_file('./known/mellisandre.jpg')
mellisandre_encoding = face_recognition.face_encodings(img_of_mellisandre)[0]

img_of_missandei = face_recognition.load_image_file('./known/missandei.jpg')
missandei_encoding = face_recognition.face_encodings(img_of_missandei)[0]

img_of_varys = face_recognition.load_image_file('./known/varys.jpg')
varys_encoding = face_recognition.face_encodings(img_of_varys)[0]


print('Files Loaded')

known_face_encodings = [
    snow_encoding,
    sam_encoding,
    arya_encoding,
    cersie_encoding,
    hound_encoding,
    brienne_encoding,
    jaime_encoding,
    sansa_encoding,
    bran_encoding,
    euron_encoding,
    gilly_encoding,
    varys_encoding,
    tyrion_encoding,
    missandei_encoding,
    mellisandre_encoding,
    daenerys_encoding,
    serdavos_encoding,
    jorah_encoding,
    greyworm_encoding,
    theon_encoding


]

known_face_names = [
    "John Snow",
    "Sam",
    "Arya",
    "Cersie",
    "Hound",
    "Brienne",
    "Jaime",
    "Sansa",
    "Bran",
    "Euron",
    "Gilly",
    "Varys",
    "Tyrion",
    "Missandei",
    "Mellisandre",
    "Daenerys",
    "Ser Davos",
    "Jorah",
    "Grey Worm",
    "Theon"
]

print('Names loaded')

img_to_detect = face_recognition.load_image_file('./unknown/s8.jpg')

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
    draw.rectangle(((left,bottom - text_height - 5), (right + 10,bottom)), fill=(0,0,0), outline=(0,0,0))
    draw.text((left + 5, bottom - text_height - 5), name, fill=(255,255,255,255))

del draw
pil_image.show()
pil_image.save('identified.jpg')
#Ignore this line, I just wrote this line just to make loc as 100! :D