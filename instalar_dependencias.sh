#!/bin/bash
# ============================================
# INSTALADOR DE DEPENDENCIAS - LINUX/MAC
# ============================================
# Script para instalar todas as bibliotecas necessarias
# Salve como: instalar_dependencias.sh
# Roda com: bash instalar_dependencias.sh

echo ""
echo "============================================"
echo "📦 INSTALADOR DE DEPENDENCIAS"
echo "============================================"
echo ""
echo "Este script vai instalar todas as bibliotecas necessarias"
echo "para rodar o Dashboard Internacoes DATASUS"
echo ""
read -p "Pressione ENTER para continuar..."

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "❌ ERRO: Python não encontrado!"
    echo ""
    echo "Para instalar Python:"
    echo "  macOS:  brew install python3"
    echo "  Ubuntu: sudo apt-get install python3"
    echo ""
    exit 1
fi

echo ""
echo "✅ Python encontrado!"
python3 --version
echo ""

# Atualizar pip
echo "📦 Atualizando pip..."
python3 -m pip install --upgrade pip

echo ""
echo "📦 Instalando bibliotecas principais..."
echo ""

# Instalar cada biblioteca
echo "Instalando Streamlit..."
pip install streamlit==1.25.0

echo "Instalando Pandas..."
pip install pandas==1.5.3

echo "Instalando NumPy..."
pip install numpy==1.24.3

echo "Instalando Plotly..."
pip install plotly==5.15.0

echo "Instalando Matplotlib..."
pip install matplotlib==3.7.1

echo "Instalando Seaborn..."
pip install seaborn==0.12.2

echo "Instalando Scikit-learn..."
pip install scikit-learn==1.2.2

echo "Instalando python-dotenv..."
pip install python-dotenv

echo "Instalando requests..."
pip install requests

echo ""
echo "============================================"
echo "✅ INSTALAÇÃO CONCLUIDA COM SUCESSO!"
echo "============================================"
echo ""
echo "Agora voce pode rodar:"
echo "  streamlit run app.py"
echo ""
echo "Seu dashboard estará em:"
echo "  http://localhost:8501"
echo ""
