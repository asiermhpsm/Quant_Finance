REGENERATE_PLOTS=false

if [ "$1" == "True" ]; then
    REGENERATE_PLOTS=true
fi

if [ "$REGENERATE_PLOTS" = true ]; then
    echo -e "[INFO]\t\tRegenerando todos los plots..."


    echo  -e "PARTE 1"


    echo  -e "[INFO]\t\tGenerando plots de los payoffs..."
    python Utils/Parte1/PayOff.py

    echo  -e "[INFO]\t\tGenerando plot de los distintos tipos de Yield Curve..."
    python Utils/Parte1/Yield_Curve.py

    echo  -e "[INFO]\t\tGenerando plots de los distintos caminos aleatorios..."
    python Utils/Parte1/Random_Walks.py

    echo  -e "[INFO]\t\tGenerando plot de la ecuación de Fokker-Planck..."
    python Utils/Parte1/Fokker_Planck.py

    echo  -e "[INFO]\t\tGenerando plot del first-time exit..."
    python Utils/Parte1/First_TIme_Exit.py

    echo  -e "[INFO]\t\tGenerando plots de las soluciones de las ecuaciones básicas de Black-Scholes..."
    python Utils/Parte1/BS_Solutions.py

    echo  -e "[INFO]\t\tGenerando plots de las estimaciones de volatilidad..."
    python Utils/Parte1/Estim_Vol.py

    echo  -e "[INFO]\t\tGenerando plot de la cobertura de la volatilidad..."
    python Utils/Parte1/Cobertura_Vol_Imp.py

    echo  -e "[INFO]\t\tGenerando plot del One-Touch call..."
    python Utils/Parte1/One_Touch.py
else
    echo -e "[INFO]\t\tNo se regenerarán los plots."
fi








echo  -e "[INFO]\t\tCompilando main.tex..."
pdflatex -draftmode -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -draftmode -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
echo  -e "[INFO]\t\tCompilación completada. El PDF está listo."


