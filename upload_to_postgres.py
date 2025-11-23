import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import ssl

# Cargar variables de entorno
load_dotenv()

def upload_data():
    try:
        # 1. Cargar tu CSV limpio
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "adult_income_clean.csv")
        
        print(f"📁 Cargando archivo: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ Archivo no encontrado: {file_path}")
            return
        
        df = pd.read_csv(file_path)
        print(f"✅ Dataset cargado: {df.shape}")
        print(f"📝 Columnas: {df.columns.tolist()}")
        
        # 2. Configurar conexión con SSL
        connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        
        # Configurar engine con SSL
        engine = create_engine(
            connection_string,
            connect_args={
                'sslmode': 'require',
                'sslrootcert': os.path.join(base_dir, 'root.crt')  # Opcional, pero recomendado
            }
        )
        
        print("🔌 Conectando a PostgreSQL con SSL...")
        
        # 3. Subir datos
        df.to_sql('adult_income_data', engine, if_exists='replace', index=False)
        print("✅ Datos subidos exitosamente!")
        
        # 4. Verificar
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM adult_income_data;"))
            count = result.scalar()
            print(f"📊 Total de registros en BD: {count}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    upload_data()