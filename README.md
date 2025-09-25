# Setting Up Development Environment (R)

- install R on your machine (Windows) https://cran-r.c3sl.ufpr.br/
- R is usually installed in this folder: C:\Program Files\R\R-4.2.3\bin . Take this folder and add it to your PATH system variable.
- install RStudio (Windows) https://posit.co/download/rstudio-desktop/
    - when asked which version of R you want to use, select the one that was locally installed in your computer by the previous two steps
- Change the RStudio theme to a dark one in Tools -> Global Options -> Appearance
- to install any packages, if the RStudio IDE doesn't show a notification already, do the following in the RStudio terminal:

``` 
install.packages("package_name")
```

- open an R script and run CTRL + SHITF + S to run it. 


# Setting Up Development Environment (Python)

- install python 3.12.4 https://www.python.org/downloads/release/python-3124/
    - make sure the python.exe is placed in the PATH during the installation wizard

- check that the installation was successful: open CMD and run `python -V`. It must show the version installed.
- clone / pull the repository
- go to folder containing project
- connect FUG and PUMP to computer. Make sure they show up in the Operating System's device manager.
- make sure the `setup.json` file has all the configurations used.

Now we need to do the following:
 * Create Pyhton virtual environment
 * Activate the virtual env
 * Install the packages in this virtual environment
 * Run the Jupyer Notebook inside this virtual environment
 
To do all those things, do the following:
- `python -m venv myEnv`
- cd to myEnv/Scripts (cd myEnv/Scripts/)
- `.\activate`
- cd back to project root folder (cd ../../)
- `pip install -r requirements.txt`
- check that all modules were installed: `pip freeze`
- choose the newly created virtual environment in the VS Code lower right corner
- run the program: `python main.py`
- to exit the virtual environment afterwards: `deactivate`


python -m pyflowchart test.py -o test.html


# Linux

 python -m venv tp1-env # Criar ambiente virtual python
  source tp1-env/bin/activate # Ativar ambiente virtual 
  pip install -r requirements.txt # Instalar dependencias
  c # Executar o servidor


# TP1 Instruções

Clique em uma célula vazia para adicionar uma parede.
Clique em uma célula ocupada para removê-la.
Segure a tecla "s" e clique em uma célula para definir o estado inicial.
Segure a tecla "g" e clique em uma célula para definir o estado final.
Segure a tecla "c" e clique em uma célula para definir o custo.

# Python Debugging

import pdb; 
pdb.set_trace()

vai abrir um terminal exatamente igual o GDB em C.