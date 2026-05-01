
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import numpy as np
import datetime
import os

app = Flask(__name__)
print("Загварыг ачаалж байна... (Loading model...)")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Загвар бэлэн боллоо! (Model ready!)")

QA_PAIRS = [
    {
        "questions": [
            "Сайн уу?", "Сайн байна уу?", "Та сайн байна уу?",
            "sain uu", "sain baina uu", "snuu", "snuu?",
            "hi", "hello", "hey", "Hi there", "Нүүр",
            "Мэнд", "мэндчилгээ", "wru", "wr"
        ],
        "answer": "Сайн уу! Би сайн байна. Танд юугаар туслах вэ? 😊"
    },
    {
        "questions": [
            "Чи хэн бэ?", "Чи юу вэ?", "Таны нэр хэн бэ?",
            "chi hen be", "tanii ner hen be", "who are you", "what are you",
            "Та хэн бэ", "намайг хэн гэж дуудах вэ"
        ],
        "answer": "Би бол Монгол хэлээр ярьдаг AI туслагч. Та надаас асуулт асуух, мэдээлэл авах боломжтой."
    },
    {
        "questions": [
            "Чи юу хийдэг вэ?", "Чи яаж ажилладаг вэ?",
            "chi yu hiideg ve", "chi yaaj ajildag ve",
            "what do you do", "how do you work",
            "Та юу хийдэг вэ", "tanai chiglel yu ve"
        ],
        "answer": "Би таны асуултад хариулах, мэдээлэл өгөх, зөвлөгөө өгөх зэрэг олон зүйл хийдэг AI туслагч. Юу мэдмээр байна?"
    },
    {
        "questions": [
            "Баяртай", "баяртай", "Баяртай!", "bayrtai","bye", "goodbye", "see you",
        ],
        "answer": "Баяртай! Дараа уулзая. Амжилт хүсье! 👋"
    },
    {
        "questions": [
            "Та сайн байна уу", "Чи яаж байна", "chi yaaj bna", "snuu yaaj bna",
            "how are you", "yu hiideg bna", "yu hiideg ve", "яаж байна"
        ],
        "answer": "Би маш сайн байна, баярлалаа! Та яаж байна вэ? 😊"
    },
    {
        "questions": [
            "Баярлалаа", "баярлалаа!", "talarkhal", "thanks", "thank you",
            "спасибо", "гялайлгасан"
        ],
        "answer": "Зүгээр ээ! Туслахдаа баяртай байна. Өөр асуулт байвал надаас асуугаарай. 😊"
    },
    {
        "questions": [
            "Уучлаарай", "уучлаарай!", "sorry", "uchlaarai", "Мартсан"
        ],
        "answer": "Зүгээр ээ, санаа зоволтгүй! Надад асуулт асуугаарай. 😊"
    },
    {
        "questions": [
            "Сургуулийн тухай юу мэдэх вэ?", "Сурах нь чухал уу?",
            "school", "education", "surguu", "surax",
            "Сурлага", "сурч байна", "bi surdag"
        ],
        "answer": "Сурлага бол амьдралын хамгийн чухал хөрөнгө оруулалт. Сайн суралцсанаар та ирээдүйдээ илүү боломж бүрдүүлж чадна. 📚"
    },
    {
        "questions": [
            "Математик хэцүү юу?", "math hezuu uu", "математик",
            "algebra", "geometry", "тооцоолол"
        ],
        "answer": "Математик эхэндээ хэцүү санагдаж болно, гэхдээ дасгал хийх тусам хялбар болдог. Өдөр бүр арай ч гэсэн дасгал хийгээрэй! ✏️"
    },
    {
        "questions": [
            "Гэрийн даалгавар", "homework", "geriin daalgavar", "HW",
            "дасгал", "даалгавар хийх"
        ],
        "answer": "Гэрийн даалгаврыг цагт нь хийх нь сурагчийн хамгийн чухал үүрэг. Хэрэв ямар нэг зүйлд бэрхшээл гарвал надаас асуугаарай! 📝"
    },
    {
        "questions": [
            "Шалгалт", "exam", "shalgalt", "шалгалтанд бэлтгэх",
            "test", "quiz", "хичээлийн шалгалт"
        ],
        "answer": "Шалгалтанд бэлтгэхийн тулд: 1️⃣ Цагаа зохицуулаарай 2️⃣ Тэмдэглэл хөтлөөрэй 3️⃣ Хангалттай унтаарай 4️⃣ Амрах завсарлага авч байгаарай. Амжилт хүсье! 💪"
    },
    {
        "questions": [
            "Монгол хэл", "mongolian language", "mongol hel",
            "Монгол цагаан толгой", "кирилл", "script"
        ],
        "answer": "Монгол хэл бол дэлхийн хамгийн онцлог хэлнүүдийн нэг. Монгол цагаан толгой нь 35 үсэгтэй бөгөөд Кирилл бичгийн тогтолцоог ашигладаг. 🇲🇳"
    },
    {
        "questions": [
            "Англи хэл сурах", "learn english", "angli hel surax",
            "english course", "how to learn english"
        ],
        "answer": "Англи хэл сурахын тулд: кино, цуврал үзэх, англи дуу сонсох, өдөр бүр 10-20 үг цээжлэх зэргийг хийгээрэй. Тогтвортой байх нь хамгийн чухал! 🌍"
    },
    {
        "questions": [
            "Технологи гэж юу вэ?", "technology", "tech", "technologi",
            "Технологийн хөгжил", "шинэ технологи"
        ],
        "answer": "Технологи бол хүний амьдралыг хялбарчилж, бүтээмжийг нэмэгдүүлдэг шинжлэх ухааны хэрэглээ. Компьютер, гар утас, интернет бүгд технологийн бүтээгдэхүүн. 💻"
    },
    {
        "questions": [
            "Интернет гэж юу вэ?", "internet", "intrnet", "net",
            "wi-fi", "wifi", "сүлжээ"
        ],
        "answer": "Интернет бол дэлхий даяар компьютер болон төхөөрөмжийг холбосон глобал сүлжээ. 1990-ээд оноос эхлэн нийтэд нээлттэй болсон. 🌐"
    },
    {
        "questions": [
            "Компьютер", "computer", "PC", "laptop", "notebook",
            "гар зөөврийн компьютер"
        ],
        "answer": "Компьютер бол тооцоолол хийх, мэдээлэл хадгалах, боловсруулах цахим төхөөрөмж. Өнөөдөр компьютергүйгээр олон салбар ажиллах боломжгүй болсон. 🖥️"
    },
    {
        "questions": [
            "Гар утас", "phone", "smartphone", "mobile", "gar utas",
            "утас", "iphone", "android", "samsung"
        ],
        "answer": "Ухаалаг гар утас нь одоо компьютерийн үүргийг гүйцэтгэдэг болсон. Монголчуудын 90%+ гар утастай. 📱"
    },
    {
        "questions": [
            "Хиймэл оюун ухаан", "artificial intelligence", "AI", "хиймэл оюун",
            "machine learning", "ML", "deep learning"
        ],
        "answer": "Хиймэл оюун ухаан (AI) бол машин дээр хүний оюун ухааны үйл ажиллагааг дуурайлган хэрэгжүүлдэг технологи. ChatGPT, Gemini, Claude зэрэг нь AI системүүд юм. 🤖"
    },
    {
        "questions": [
            "Программчлал гэж юу вэ?", "programming", "coding", "code",
            "programmchlal", "кодинг", "програм бичих"
        ],
        "answer": "Программчлал бол компьютерт тушаал өгөх зааврын хэл. Python, JavaScript, Java зэрэг программчлалын хэлнүүд байдаг. 🖥️"
    },
    {
        "questions": [
            "Python гэж юу вэ?", "python", "python хэл", "Python programming",
            "learn python", "python surax"
        ],
        "answer": "Python бол энгийн, уншихад хялбар, олон зорилгоор ашиглаж болох программчлалын хэл. Data Science, AI, вэб хөгжүүлэлтэд өргөн хэрэглэгддэг. 🐍"
    },
    {
        "questions": [
            "JavaScript гэж юу вэ?", "javascript", "JS", "js хэл",
            "web development", "frontend"
        ],
        "answer": "JavaScript бол вэб хөгжүүлэлтийн хамгийн өргөн хэрэглэгддэг хэл. Браузерт ажилладаг цорын ганц программчлалын хэл юм. 🌐"
    },
    {
        "questions": [
            "HTML гэж юу вэ?", "html", "HTML5", "вэб хуудас",
            "web page", "webpage"
        ],
        "answer": "HTML (HyperText Markup Language) бол вэб хуудасны бүтцийг тодорхойлдог markup хэл. Бүх вэб сайт HTML-ийн суурьтай байдаг. 📄"
    },
    {
        "questions": [
            "CSS гэж юу вэ?", "css", "CSS3", "style", "дизайн",
            "загвар", "stylesheet"
        ],
        "answer": "CSS (Cascading Style Sheets) бол вэб хуудасны харагдах байдлыг тодорхойлдог хэл. Өнгө, фонт, байрлал зэргийг CSS-ээр тохируулдаг. 🎨"
    },
    {
        "questions": [
            "Database гэж юу вэ?", "database", "DB", "SQL", "мэдээллийн сан",
            "sqlite", "mysql", "postgresql"
        ],
        "answer": "Мэдээллийн сан (Database) бол мэдээллийг зохион байгуулалттай хадгалах систем. SQLite, MySQL, PostgreSQL зэрэг нь алдартай мэдээллийн сангийн системүүд. 🗄️"
    },
    {
        "questions": [
            "API гэж юу вэ?", "api", "REST API", "endpoint",
            "backend", "серверийн тал"
        ],
        "answer": "API (Application Programming Interface) бол программ хооронд харилцах интерфэйс. Жишээлбэл, цаг агаарын апп нь цаг агаарын серверийн API-г ашиглан мэдээлэл авдаг. 🔗"
    },
    {
        "questions": [
            "Git гэж юу вэ?", "git", "github", "version control",
            "хувилбарын удирдлага"
        ],
        "answer": "Git бол кодын хувилбарыг удирдах систем. GitHub бол Git-ийн дээр тулгуурласан алдартай платформ бөгөөд хөгжүүлэгчид кодоо хадгалж, хамтрал хийдэг. 📦"
    },
    {
        "questions": [
            "Flask гэж юу вэ?", "flask", "flask python", "web framework",
            "django", "fastapi"
        ],
        "answer": "Flask бол Python дээр суурилсан хөнгөн вэб framework. Энгийн бөгөөд шинэ хөгжүүлэгчдэд тохиромжтой. Django бол илүү том, бүрэн функцтэй framework юм. ⚗️"
    },
    {
        "questions": [
            "Программист болохын тулд яах вэ?", "how to become programmer",
            "programmist boloh", "кодинг сурах", "dev boloh"
        ],
        "answer": "Программист болохын тулд: 1️⃣ Нэг хэл сонгоорой (Python санал болгож байна) 2️⃣ Өдөр бүр дасгал хийгээрэй 3️⃣ Жижиг төслүүд хийгээрэй 4️⃣ Stack Overflow, GitHub ашиглаарай 5️⃣ Тасралтгүй суралцаарай! 💻"
    },
    {
        "questions": [
            "Өнөөдөр ямар цаг вэ?", "цаг хэд болж байна", "what time is it",
            "tsag hed boljbaina", "хэдэн цаг"
        ],
        "answer": "Харамсалтай нь би цаг мэдэхгүй байна, учир нь интернэтгүй ажилладаг. Та утасны цагаасаа харж болно! 🕐"
    },
    {
        "questions": [
            "Өнөөдөр ямар өдөр вэ?", "what day is it", "date",
            "odoodor yamar odor ve", "хэдний өдөр"
        ],
        "answer": "Харамсалтай нь би огноо мэдэхгүй. Та утасны календариасаа харж болно! 📅"
    },
    {
        "questions": [
            "Цаг агаар ямар байна?", "weather", "tsag agaar", "уур амьсгал",
            "бороо", "цас", "нар"
        ],
        "answer": "Харамсалтай нь би цаг агаарын мэдээ мэдэхгүй. Google дээр 'Улаанбаатарын цаг агаар' гэж хайж болно! ☀️"
    },
    {
        "questions": [
            "Хоол идэж байна уу?", "хоол", "food", "хоолны жор",
            "recipe", "mongolian food", "монгол хоол"
        ],
        "answer": "Монгол хоол маш амттай! Хуушуур, банш, гурилтай шөл, цуйван зэрэг нь алдартай Монгол хоолнууд. Ямар хоол хиймээр байна? 🥟"
    },
    {
        "questions": [
            "Уснаас архи уух уу?", "ундаа", "drink", "цай",
            "сүүтэй цай", "airag", "айраг", "цагаан идээ"
        ],
        "answer": "Монгол цагаан идээ дотроос айраг, тараг, аарц, ааруул зэрэг нь эрүүл, уламжлалт хүнс. Айраг бол Монголчуудын үндэсний ундаа. 🍵"
    },
    {
        "questions": [
            "Унтах хэрэгтэй юу?", "sleep", "унтах", "нойр",
            "нойрсох", "insomnia", "нойргүй"
        ],
        "answer": "Эрүүл унтах нь маш чухал! Насанд хүрсэн хүн өдөрт 7-9 цаг унтах хэрэгтэй. Нойрны дутагдал нь анхаарал, бүтээмжийг бууруулдаг. 😴"
    },
    {
        "questions": [
            "Дасгал хийх хэрэгтэй юу?", "exercise", "sport", "gym",
            "тамирлах", "спорт", "фитнесс"
        ],
        "answer": "Тийм, дасгал хийх нь биеийн эрүүл мэнд, сэтгэлийн тайван байдалд маш сайн нөлөөтэй. Өдөрт 30 минутын дасгал хийхийг зорьоорой. 💪"
    },
    {
        "questions": [
            "Эрүүл мэнд", "health", "eruul mend", "эмч",
            "doctor", "эмнэлэг", "hospital"
        ],
        "answer": "Эрүүл мэндийг сахихын тулд: тогтмол дасгал хийх, эрүүл хоол идэх, хангалттай унтах, стрессийг удирдах нь чухал. Жилд нэг удаа эмчид үзүүлж байгаарай. 🏥"
    },
    {
        "questions": [
            "ChatGPT гэж юу вэ?", "chatgpt", "GPT", "openai",
            "chat gpt", "gpt4"
        ],
        "answer": "ChatGPT бол OpenAI компанийн хөгжүүлсэн том хэлний загвар дээр суурилсан AI туслагч. 2022 оны 11 сард нийтэд нээгдэж маш хурдан алдаршсан. 🤖"
    },
    {
        "questions": [
            "Claude гэж юу вэ?", "claude", "anthropic",
            "claude ai", "claude chatbot"
        ],
        "answer": "Claude бол Anthropic компанийн хөгжүүлсэн AI туслагч. Аюулгүй, найдвартай AI бүтээхэд чиглэсэн Anthropic компанийн гол бүтээгдэхүүн. 🧠"
    },
    {
        "questions": [
            "Gemini гэж юу вэ?", "gemini", "google ai", "bard",
            "google gemini"
        ],
        "answer": "Gemini бол Google-ийн хөгжүүлсэн олон модаль AI загвар. Текст, зураг, дуу, видео зэргийг ойлгож, боловсруулах чадвартай. ✨"
    },
    {
        "questions": [
            "AI аюулгүй юу?", "AI dangerous", "AI аюул", "ai ayultai yuu",
            "аюулгүй AI"
        ],
        "answer": "AI технологи зөв хэрэглэвэл маш хэрэгтэй. Гэхдээ буруу зорилгоор ашиглаж болзошгүй тул AI-ийн ёс зүй, аюулгүй байдлын судалгаа чухал. 🛡️"
    },
    {
        "questions": [
            "Machine learning гэж юу вэ?", "machine learning", "ML",
            "нейрон сүлжээ", "neural network", "deep learning"
        ],
        "answer": "Machine Learning бол машинд өгөгдөлд суурилан суралцах чадвар олгодог AI-ийн салбар. Нейрон сүлжээ нь хүний тархины бүтцийг загварчилсан ML арга. 🧬"
    },
    {
        "questions": [
            "Вэб сайт хэрхэн бүтээх вэ?", "how to make website",
            "web site hiih", "вэб хөгжүүлэлт", "web development"
        ],
        "answer": "Вэб сайт бүтээхийн тулд: 1️⃣ HTML (бүтэц) 2️⃣ CSS (загвар) 3️⃣ JavaScript (функц) сурах хэрэгтэй. React, Vue, Angular зэрэг framework-уудыг дараа нь сурж болно. 🌐"
    },
    {
        "questions": [
            "React гэж юу вэ?", "react", "reactjs", "react.js",
            "react framework", "jsx"
        ],
        "answer": "React бол Facebook (Meta)-ийн хөгжүүлсэн JavaScript library. Динамик, хурдан вэб апп бүтээхэд хэрэглэгддэг бөгөөд өнөөдөр хамгийн алдартай frontend хэрэгслүүдийн нэг. ⚛️"
    },
    {
        "questions": [
            "Hosting гэж юу вэ?", "hosting", "server", "deployment",
            "deploy", "вэб сайт байршуулах"
        ],
        "answer": "Hosting бол таны вэб сайтыг интернэтэд байршуулах үйлчилгээ. Vercel, Netlify (үнэгүй), AWS, DigitalOcean зэрэг нь алдартай hosting үйлчилгээнүүд. ☁️"
    },
    {
        "questions": [
            "Domain гэж юу вэ?", "domain", "домэйн нэр",
            "url", "website address", "www"
        ],
        "answer": "Domain бол вэб сайтын хаяг, жишээ нь google.com. .mn домэйн бол Монгол Улсын домэйн. GoDaddy, Namecheap зэрэг сайтаас domain худалдаж авч болно. 🌍"
    },
    {
        "questions": [
            "Амжилтын нууц юу вэ?", "success", "amjilt", "амжилттай болох",
            "how to be successful", "амжилт олох"
        ],
        "answer": "Амжилтын нууц: 🎯 Зорилго тодорхой тавих, 📚 Тасралтгүй суралцах, 💪 Тэвчээртэй байх, 🤝 Сайн хүмүүстэй найзлах, ❌ Алдаанаасаа суралцах. Та чадна!"
    },
    {
        "questions": [
            "Цаг хугацааг хэрхэн зохицуулах вэ?", "time management",
            "tsag zorichuulah", "хугацаа удирдах", "productivity"
        ],
        "answer": "Цаг зохицуулах зөвлөмж: ✅ To-do жагсаалт хийгээрэй, ⏰ Pomodoro арга хэрэглэ (25 мин ажил, 5 мин амрах), 📵 Сэтгэл татахуйц зүйлсийг хязгаарлаарай, 🎯 Хамгийн чухал зүйлээ эхэлж хийгээрэй."
    },
    {
        "questions": [
            "Стресстэй байна", "stressed", "stres", "стресс",
            "anxiety", "worried", "санаа зоволт", "сэтгэл хямрал"
        ],
        "answer": "Стресстэй байхад: 🧘 Гүн амьсгалаарай, 🚶 Богино зугаалгаарай, 💬 Найздаа хэлээрэй, 📖 Дуртай зүйлдээ анхаараарай. Бүх зүйл сайхан болно! 💙"
    },
    {
        "questions": [
            "Найз нөхөд олох арга", "how to make friends", "найз хийх",
            "social skills", "нийгмийн ур чадвар"
        ],
        "answer": "Найз олохын тулд: 😊 Нааштай инээмсэглэлтэй байгаарай, 👂 Сайн сонсогч байгаарай, 🤝 Сонирхол нийлэх хүмүүстэй нийлэгтэй, 💬 Эхлэн ярьж эхлэх зоригтой байгаарай!"
    },
    {
        "questions": [
            "Мөнгө хэрхэн хэмнэх вэ?", "save money", "mungu hemnel",
            "хэмнэлт", "budget", "sankhuu"
        ],
        "answer": "Мөнгө хэмнэх аргууд: 📊 Зардлаа бүртгэгтэй, 🛒 Хүнсний дэлгүүрт жагсаалттай явж, ☕ Гадуур кофе ууж хэт зарцуулалгүй, 💰 Орлогынхоо 20% хадгаламжид хийгтэй."
    },
    {
        "questions": [
            "Аялал", "travel", "ayalal", "дотоодын аялал",
            "Монголоор аялах", "tourist"
        ],
        "answer": "Монгол бол аялалын гайхалтай газар! 🏔️ Хөвсгөл нуур, Говь цөл, Орхоны хөндий, Тэрэлж зэрэг нь алдартай аялалын цэгүүд. Монгол нутаг дэлхийн хамгийн их ногоон нутаг бусдыг уриасай!"
    },
    {
        "questions": [
            "Монгол улс хаана байдаг вэ?", "where is mongolia",
            "mongolia", "Монгол", "Монгол улс"
        ],
        "answer": "Монгол улс нь Төв Азид, Хятад болон Оросын хооронд оршдог. Нийслэл нь Улаанбаатар хот. Дэлхийн хамгийн том газар нутагтай 19-р улс. 🇲🇳"
    },
    {
        "questions": [
            "Улаанбаатар", "ulaanbaatar", "UB", "нийслэл",
            "capital city"
        ],
        "answer": "Улаанбаатар нь Монгол улсын нийслэл бөгөөд улсын хүн амын 50% гаруй нь амьдардаг хамгийн том хот. Дэлхийн хамгийн хүйтэн нийслэл хотуудын нэг. 🏙️"
    },
    {
        "questions": [
            "Чингис хаан", "Genghis Khan", "chingis haan",
            "Монгол эзэнт гүрэн", "mongol empire"
        ],
        "answer": "Чингис хаан (1162-1227) бол Монгол эзэнт гүрнийг байгуулсан агуу жанжин. Тэрээр дэлхийн түүхэнд хамгийн том газар нутгийг нэгтгэн захирсан. 🏹"
    },
    {
        "questions": [
            "Монголын хүн ам", "population of mongolia",
            "mongol hun am", "хэдэн хүн амьдардаг"
        ],
        "answer": "Монгол улсын хүн ам 2024 онд ойролцоогоор 3.4 сая байна. Нийт газар нутаг нь 1,564,116 км² бөгөөд хамгийн бага хүн амын нягтралтай улсуудын нэг. 📊"
    },
    {
        "questions": [
            "Онигоо хэлж өг", "tell me a joke", "joke", "хөгжилтэй зүйл",
            "инээд", "хошин шог"
        ],
        "answer": "Программист яагаад гэрлэхгүй байдаг вэ? ...Учир нь хаанаас ч эхлэсэн null pointer exception гардаг! 😄 (Программистуудад ойлгогдоно 😅)"
    },
    {
        "questions": [
            "Дуртай өнгө", "favorite color", "color", "өнгө",
            "ямар өнгө дуртай вэ"
        ],
        "answer": "Би AI учраас өнгийг харж чадахгүй, гэхдээ Монголын цэнхэр тэнгэр маш гоё гэж сонссон! 🔵 Таны дуртай өнгө юу вэ?"
    },
    {
        "questions": [
            "Дуртай хоол", "favorite food", "алинаас нь дуртай",
            "хамгийн амттай хоол"
        ],
        "answer": "Би хоол идэж чадахгүй AI ч гэлээ, монгол хуушуур хамгийн амттай гэж мэднэ! 🥟 Таны дуртай хоол юу вэ?"
    },
    {
        "questions": [
            "Надтай нөхөрлөх үү?", "be my friend", "найз болох уу",
            "наад", "bff"
        ],
        "answer": "Мэдээж! Би таны AI найз болохдоо баяртай байна. Хэзээ ч асуулт асуух, ярилцах боломжтой. 😊🤝"
    },
    {
        "questions": [
            "Чи мэдрэмж байдаг уу?", "do you have feelings", "chi medremj baidagtai yuu",
            "чи сэтгэл мэдэрдэг үү", "emotions"
        ],
        "answer": "Би AI учраас хүний мэдрэмж шиг мэдрэлгүй. Гэхдээ таны асуултад хариулахад 'хэрэгтэй' гэсэн зорилго надад байдаг! 🤖💙"
    },
    {
        "questions": [
            "Хамгийн сайн AI хэн бэ?", "best AI", "hamgiin sain ai",
            "top AI", "ChatGPT vs Claude vs Gemini"
        ],
        "answer": "ChatGPT, Claude, Gemini зэрэг нь бүгд гайхалтай AI системүүд. Хэрэглэх зорилгоосоо хамааран аль нь тохиромжтойг сонгох нь зөв. Миний хувьд Монгол хэлийг дэмжиж чадаж байна! 😄"
    },
]
print("Асуултуудын embedding тооцоолж байна...")
dataset = []
for pair in QA_PAIRS:
    for question in pair["questions"]:
        embedding = model.encode(question)
        dataset.append({
            "question": question,
            "answer": pair["answer"],
            "embedding": embedding,
        })
print(f"Нийт {len(dataset)} асуулт бэлэн болсон!")
DB_PATH = "chat_history.db"

def init_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_msg  TEXT    NOT NULL,
            bot_resp  TEXT    NOT NULL,
            timestamp TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_message(user_msg, bot_resp):
    """Save a conversation turn to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO messages (user_msg, bot_resp, timestamp) VALUES (?, ?, ?)",
        (user_msg, bot_resp, ts)
    )
    conn.commit()
    conn.close()

def load_history():
    """Load the last 30 conversation turns."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT user_msg, bot_resp, timestamp FROM messages ORDER BY id DESC LIMIT 30"
    )
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))  

SIMILARITY_THRESHOLD = 0.50  
FALLBACK = "Уучлаарай, би таны асуултыг ойлгосонгүй. Өөрөөр асуугаад үзнэ үү. 🙏"

def get_answer(user_text):
    """Encode user message and find the best matching answer."""
    user_embedding = model.encode(user_text)
    all_embeddings = np.array([d["embedding"] for d in dataset])
  
    scores = cosine_similarity([user_embedding], all_embeddings)[0]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= SIMILARITY_THRESHOLD:
        return dataset[best_idx]["answer"], best_score
    return FALLBACK, best_score

@app.route("/")
def index():
    history = load_history()
    return render_template("index.html", history=history)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Хоосон мэдэгдэл"}), 400

    answer, score = get_answer(user_msg)
    save_message(user_msg, answer)
    return jsonify({"response": answer, "score": round(score, 3)})

@app.route("/clear", methods=["POST"])
def clear():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages")
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    init_db()
    print("\n server http://127.0.0.1:5000 \n")
    app.run(debug=True, port=5000)

