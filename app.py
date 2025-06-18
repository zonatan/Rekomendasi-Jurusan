from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Daftar jurusan
JURUSAN = [
    'TEKNIK ELEKTRO', 'TEKNIK INDUSTRI', 'TEKNIK INFORMATIKA', 'TEKNIK SIPIL',
    'AKUNTANSI', 'MANAJEMEN', 'ILMU KEPERAWATAN', 'PROFESI NERS',
    'FISIOTERAPI', 'AGRIBISNIS', 'ILMU HUKUM', 'HOSPITALITY DAN PARIWISATA',
    'PENDIDIKAN GURU SD'
]

# Kriteria relevan per jurusan
KRITERIA_JURUSAN = {
    'TEKNIK ELEKTRO': [
        'Suka_Angka', 'Mapel_Favorit_IPA', 'Tertarik_Teknologi', 'Suka_Proyek',
        'Suka_Masalah_Teknis', 'Cita_Teknologi', 'Kemampuan_Analisis', 'Ketelitian'
    ],
    'TEKNIK INDUSTRI': [
        'Suka_Angka', 'Mapel_Favorit_IPA', 'Tertarik_Teknologi', 'Tertarik_Bisnis',
        'Suka_Proyek', 'Suka_Masalah_Teknis', 'Kemampuan_Analisis', 'Suka_Kerja_Tim'
    ],
    'TEKNIK INFORMATIKA': [
        'Suka_Angka', 'Mapel_Favorit_IPA', 'Tertarik_Teknologi', 'Suka_Proyek',
        'Suka_Masalah_Teknis', 'Cita_Teknologi', 'Kemampuan_Analisis', 'Ketelitian', 'Ekstra_IT'
    ],
    'TEKNIK SIPIL': [
        'Suka_Angka', 'Mapel_Favorit_IPA', 'Tertarik_Teknologi', 'Suka_Proyek',
        'Suka_Masalah_Teknis', 'Kemampuan_Analisis', 'Ketelitian', 'Suka_Kerja_Tim'
    ],
    'AKUNTANSI': [
        'Suka_Angka', 'Mapel_Favorit_IPS', 'Tertarik_Bisnis', 'Kemampuan_Analisis',
        'Ketelitian', 'Pengalaman_Organisasi'
    ],
    'MANAJEMEN': [
        'Suka_Teori', 'Tertarik_Bisnis', 'Suka_Public_Speaking', 'Suka_Kerja_Tim',
        'Cita_Pengusaha', 'Komunikasi', 'Pengalaman_Organisasi'
    ],
    'ILMU KEPERAWATAN': [
        'Mapel_Favorit_IPA', 'Tertarik_Kesehatan', 'Suka_Kerja_Tim', 'Cita_Kesehatan',
        'Ketelitian', 'Komunikasi'
    ],
    'PROFESI NERS': [
        'Mapel_Favorit_IPA', 'Tertarik_Kesehatan', 'Suka_Kerja_Tim', 'Cita_Kesehatan',
        'Ketelitian', 'Komunikasi'
    ],
    'FISIOTERAPI': [
        'Mapel_Favorit_IPA', 'Tertarik_Kesehatan', 'Suka_Kerja_Tim', 'Cita_Kesehatan',
        'Ketelitian'
    ],
    'AGRIBISNIS': [
        'Suka_Angka', 'Mapel_Favorit_IPS', 'Tertarik_Bisnis', 'Suka_Proyek',
        'Cita_Pengusaha', 'Kemampuan_Analisis'
    ],
    'ILMU HUKUM': [
        'Suka_Teori', 'Mapel_Favorit_IPS', 'Tertarik_Hukum', 'Suka_Menulis',
        'Suka_Public_Speaking', 'Kemampuan_Analisis', 'Ekstra_Debat'
    ],
    'HOSPITALITY DAN PARIWISATA': [
        'Suka_Public_Speaking', 'Suka_Kerja_Tim', 'Komunikasi', 'Kreativitas',
        'Pengalaman_Organisasi'
    ],
    'PENDIDIKAN GURU SD': [
        'Suka_Teori', 'Mapel_Favorit_IPS', 'Suka_Menulis', 'Suka_Public_Speaking',
        'Cita_Pendidikan', 'Komunikasi', 'Ketelitian'
    ]
}

# Daftar semua fitur
FEATURES = [
    'Suka_Teori', 'Suka_Angka', 'Suka_Seni', 'Suka_Public_Speaking', 'Suka_Menulis',
    'Mapel_Favorit_IPA', 'Mapel_Favorit_IPS', 'Tertarik_Teknologi', 'Tertarik_Bisnis',
    'Tertarik_Kesehatan', 'Tertarik_Hukum', 'Suka_Kerja_Tim', 'Suka_Proyek',
    'Suka_Masalah_Teknis', 'Cita_Pengusaha', 'Cita_Teknologi', 'Cita_Kesehatan',
    'Cita_Pendidikan', 'Kemampuan_Analisis', 'Ketelitian', 'Kreativitas', 'Komunikasi',
    'Ekstra_IT', 'Ekstra_Debat', 'Pengalaman_Organisasi'
]


def generate_synthetic_data(n_samples=100):
    data = []
    for _ in range(n_samples):
        jurusan = np.random.choice(JURUSAN)
        row = {}
        criteria = KRITERIA_JURUSAN[jurusan]
        
        for feature in FEATURES:
            if feature in criteria:
                row[feature] = np.random.choice([1, 0], p=[0.8, 0.2])
            else:
                row[feature] = np.random.choice([1, 0], p=[0.2, 0.8])
        row['Jurusan'] = jurusan
        data.append(row)
    return pd.DataFrame(data)

# Train Decision Tree model
logger.info("Generating synthetic dataset and training Decision Tree model")
df = generate_synthetic_data()
X = df.drop('Jurusan', axis=1)
y = df['Jurusan']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
logger.info(f"Model classes: {model.classes_}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        input_data = request.json
        logger.info(f"Received input: {input_data}")
        
        # Validate input
        if not input_data:
            logger.error("No input data provided")
            return jsonify({'success': False, 'error': 'No input data provided'})
        
        if not all(feature.lower() in input_data for feature in FEATURES):
            missing_features = [f for f in FEATURES if f.lower() not in input_data]
            logger.error(f"Missing features: {missing_features}")
            return jsonify({'success': False, 'error': f'Missing required features: {missing_features}'})
        
        # Prepare user input
        user_input = {}
        for feature in FEATURES:
            value = input_data.get(feature.lower(), 0)
            if isinstance(value, str):
                value = value.lower() in ['ya', 'yes', 'true', '1', 'y']
            user_input[feature] = 1 if value else 0
        logger.info(f"Processed input: {user_input}")
        
       
        input_df = pd.DataFrame([user_input], columns=FEATURES)
        probabilities = model.predict_proba(input_df)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        logger.info(f"Prediction probabilities: {prob_dict}")
        
      
        recommendations = []
        for jurusan in JURUSAN:
            criteria = KRITERIA_JURUSAN[jurusan]
            matched_criteria = [f for f in criteria if user_input.get(f, 0) == 1]
            criteria_match = len(matched_criteria) / len(criteria)
            tree_probability = prob_dict.get(jurusan, 0)
           
            match_percentage = (criteria_match * 0.6 + tree_probability * 0.4) * 100
            match_percentage = round(match_percentage, 1)
            
            recommendations.append({
                'jurusan': jurusan,
                'match_percentage': match_percentage,
                'matched_criteria': matched_criteria,
                'description': get_jurusan_description(jurusan)
            })
        
        
        recommendations = sorted(recommendations, key=lambda x: x['match_percentage'], reverse=True)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'input_features': user_input
        })
    except Exception as e:
        logger.error(f"Error processing recommendation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Terjadi kesalahan dalam memproses rekomendasi'
        })

def get_jurusan_description(jurusan):
    
    descriptions = {
        'TEKNIK ELEKTRO': 'Jurusan yang mempelajari tentang listrik, elektronika, dan sistem tenaga.',
        'TEKNIK INDUSTRI': 'Jurusan yang menggabungkan teknik dan manajemen untuk optimasi sistem produksi.',
        'TEKNIK INFORMATIKA': 'Jurusan yang fokus pada pemrograman, algoritma, dan pengembangan perangkat lunak.',
        'TEKNIK SIPIL': 'Jurusan yang berfokus pada perancangan dan pembangunan infrastruktur.',
        'AKUNTANSI': 'Jurusan yang fokus pada pencatatan dan analisis keuangan.',
        'MANAJEMEN': 'Jurusan yang mempelajari pengelolaan bisnis dan organisasi.',
        'ILMU KEPERAWATAN': 'Jurusan yang mempersiapkan siswa untuk menjadi perawat profesional.',
        'PROFESI NERS': 'Program lanjutan untuk menjadi perawat dengan keahlian klinis yang lebih mendalam.',
        'FISIOTERAPI': 'Jurusan yang fokus pada rehabilitasi fisik dan terapi gerakan.',
        'AGRIBISNIS': 'Jurusan yang menggabungkan pertanian dengan manajemen bisnis.',
        'ILMU HUKUM': 'Jurusan yang mempelajari sistem hukum dan perundang-undangan.',
        'HOSPITALITY DAN PARIWISATA': 'Jurusan yang mempelajari manajemen perhotelan dan industri pariwisata.',
        'PENDIDIKAN GURU SD': 'Jurusan yang mempersiapkan siswa untuk menjadi guru sekolah dasar.'
    }
    return descriptions.get(jurusan, 'Deskripsi jurusan tidak tersedia.')

@app.route('/features', methods=['GET'])
def get_features():
    
    features_list = [{
        'name': feature,
        'question': generate_question(feature),
        'type': 'boolean'
    } for feature in FEATURES]
    
    return jsonify({
        'success': True,
        'features': features_list
    })

def generate_question(feature_name):
    
    questions = {
        'Suka_Teori': 'Apakah Anda menyukai pembelajaran teoritis?',
        'Suka_Angka': 'Apakah Anda menyukai pekerjaan yang melibatkan angka?',
        'Suka_Seni': 'Apakah Anda memiliki minat dalam seni dan kreativitas visual?',
        'Suka_Public_Speaking': 'Apakah Anda nyaman berbicara di depan umum?',
        'Suka_Menulis': 'Apakah Anda menikmati kegiatan menulis?',
        'Mapel_Favorit_IPA': 'Apakah mata pelajaran IPA (Fisika, Kimia, Biologi) favorit Anda?',
        'Mapel_Favorit_IPS': 'Apakah mata pelajaran IPS (Ekonomi, Sosiologi, Geografi) favorit Anda?',
        'Tertarik_Teknologi': 'Apakah Anda tertarik dengan perkembangan teknologi?',
        'Tertarik_Bisnis': 'Apakah Anda tertarik dengan dunia bisnis dan entrepreneurship?',
        'Tertarik_Kesehatan': 'Apakah Anda tertarik dengan bidang kesehatan?',
        'Tertarik_Hukum': 'Apakah Anda tertarik dengan bidang hukum dan perundangan?',
        'Suka_Kerja_Tim': 'Apakah Anda menyukai bekerja dalam tim?',
        'Suka_Proyek': 'Apakah Anda menyukai pekerjaan berbasis proyek?',
        'Suka_Masalah_Teknis': 'Apakah Anda menikmati memecahkan masalah teknis?',
        'Cita_Pengusaha': 'Apakah Anda bercita-cita menjadi pengusaha?',
        'Cita_Teknologi': 'Apakah Anda bercita-cita bekerja di bidang teknologi?',
        'Cita_Kesehatan': 'Apakah Anda bercita-cita bekerja di bidang kesehatan?',
        'Cita_Pendidikan': 'Apakah Anda bercita-cita bekerja di bidang pendidikan?',
        'Kemampuan_Analisis': 'Apakah Anda memiliki kemampuan analisis yang baik?',
        'Ketelitian': 'Apakah Anda orang yang teliti dalam bekerja?',
        'Kreativitas': 'Apakah Anda menganggap diri Anda sebagai orang yang kreatif?',
        'Komunikasi': 'Apakah Anda memiliki kemampuan komunikasi yang baik?',
        'Ekstra_IT': 'Apakah Anda aktif dalam kegiatan ekstrakurikuler terkait IT?',
        'Ekstra_Debat': 'Apakah Anda aktif dalam kegiatan debat atau public speaking?',
        'Pengalaman_Organisasi': 'Apakah Anda memiliki pengalaman organisasi?'
    }
    return questions.get(feature_name, f"Apakah Anda memiliki karakteristik {feature_name.replace('_', ' ').lower()}?")

if __name__ == '__main__':
    app.run(debug=True)