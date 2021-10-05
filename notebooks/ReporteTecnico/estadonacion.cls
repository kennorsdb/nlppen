\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{estadonacion}[2020/07/06 Plantilla para el Estado de la Nacion]

% -------------------------------------------------------------------------------------------------------------------------------------
% Paquetes
% -------------------------------------------------------------------------------------------------------------------------------------

% Manejo de espacio y márgenes
\LoadClass[onecolumn]{report}
\RequirePackage[top=2.5cm,bottom=2.5cm,right=2.5cm,left=2.5cm,paper=letterpaper]{geometry}
\RequirePackage{fancyhdr}

% Opciones de página
\RequirePackage{changepage} 
\RequirePackage{setspace}
\RequirePackage{graphbox}


% Gráficos y colores
\RequirePackage{xcolor}
\RequirePackage{graphicx} 
\usepackage[singlelinecheck=false,labelsep=newline,font=bf,labelfont=bf,skip=2pt,justification=raggedright]{caption}

% Fuente
\RequirePackage{plex-serif}
\RequirePackage{plex-mono}
\RequirePackage[sfdefault, sb]{plex-sans}

% tablas
\RequirePackage{booktabs}
\RequirePackage{float}
% Las tablas siempre aparecen junto al lugar donde se colocan
\floatplacement{table}{H}


% -------------------------------------------------------------------------------------------------------------------------------------
% Declaraciones
% -------------------------------------------------------------------------------------------------------------------------------------

% Declaración de colores

% Negros y grises
\definecolor{negro}{RGB}{31,30,30} 
\definecolor{gris1}{RGB}{75,85,90} 
\definecolor{gris2}{RGB}{117,126,132} 
\definecolor{gris3}{RGB}{157,166,171} 
\definecolor{gris4}{RGB}{181,194,201} 
\definecolor{gris5}{RGB}{206,217,98}  
\definecolor{gris6}{RGB}{219,227,231} 
%otros colores
\definecolor{AzulINT1}{RGB}{0,72,135} 

% Declaración de Argumentos de la Clase
\newcommand{\nombreinforme}{Programa Estado de la Nación}
\newcommand{\imagenportada}{pen-portada.png}
\definecolor{color1}{RGB}{245,209,30}
\definecolor{color2}{RGB}{35,78,144}

\DeclareOption{EN}  {\definecolor{color1}{RGB}{35,78,144}
                        \definecolor{color2}{RGB}{245,209,30}
                        \renewcommand{\nombreinforme}{Informe Estado de la Nación en Desarrollo Humano Sostenible}
                        \renewcommand{\imagenportada}{estadonacion/en-portada.png}
}

\DeclareOption{EJ}  {\definecolor{color1}{RGB}{137,156,174}
                        \definecolor{color2}{RGB}{31,30,30}
                        \renewcommand{\nombreinforme}{Informe Estado de la Justicia}
                        \renewcommand{\imagenportada}{estadonacion/ej-portada.png}
}
                        
\DeclareOption{EE}  {\definecolor{color1}{RGB}{104,173,223}
                        \definecolor{color2}{RGB}{35,78,144}
                        \renewcommand{\nombreinforme}{Informe Estado de la Educación}
                        \renewcommand{\imagenportada}{estadonacion/ee-portada.png}
}
                        
\DeclareOption{ECCTI}  {\definecolor{color1}{RGB}{154,188,59}
                        \definecolor{color2}{RGB}{61,102,170}
                        \renewcommand{\nombreinforme}{Informe Estado de las Capacidades en Ciencia, Tecnología e Innovación}
                         \renewcommand{\imagenportada}{estadonacion/eccti-portada.png}
}
                        
\DeclareOption{ERCA}  {\definecolor{color1}{RGB}{218,192,112}
                        \definecolor{color2}{RGB}{218,142,34}
                        \renewcommand{\nombreinforme}{Informe Estado de la Región}
                        \renewcommand{\imagenportada}{estadonacion/ercaPortada.png}
    
}


\ProcessOptions\relax

% Español
\usepackage[spanish]{babel}

\newcommand{\headlinecolor}{\normalcolor}

\RequirePackage{xcolor}
\definecolor{slcolor}{HTML}{882B21}

%Definición de variables
\def\antetitulo#1{\gdef\@antetitulo{#1}} % Antetitulo
\def\address#1{\gdef\@address{#1}} % Antetitulo
\def\tipodocumento#1{\gdef\@tipodocumento{#1}} % Antetitulo
\def\annoinforme#1{\gdef\@annoinforme{#1}} % Antetitulo

\newcommand{\autoria}[2]{\gdef\@calidadautor{#1} \gdef\@author{#2} } % Antetitulo


% -------------------------------------------------------------------------------------------------------------------------------------
% Encabezados y pie de página
% -------------------------------------------------------------------------------------------------------------------------------------
\pagestyle{fancy}
\fancyhf{}
\rhead{}
\lhead{{\textit \@title}}
\rfoot{\thepage}
\renewcommand{\footrulewidth}{0.5pt}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\headlinecolor}{\black}


% -------------------------------------------------------------------------------------------------------------------------------------
% Portada
% -------------------------------------------------------------------------------------------------------------------------------------
\renewcommand{\maketitle}{%
    % Cambio de márgenes para la portada
    \newgeometry{top=1.5cm,bottom=1.5cm,right=1cm,left=1cm}
    \thispagestyle{empty}
    \parskip=0em
    
    % Imagen del encabezado
    \protect \noindent \includegraphics[width=\textwidth]{\imagenportada}
    
    % Nombre del Informe y líneas
    \noindent \textcolor{gris1}{\rule{\textwidth}{1pt}} \par
    \begin{adjustwidth}{2.0cm}{0.5cm}
        {\color{AzulINT1} \parskip=0.3cm \parindent=3cm \fontseries{sb} {\fontsize{14}{100}\selectfont \nombreinforme} \par}
    \end{adjustwidth}
    \noindent \textcolor{gris1}{\rule{\textwidth}{1pt}} \par
    
    % Tipo de documento
    \begin{adjustwidth}{2.0cm}{0.0cm}
        {\color{AzulINT1} \parskip=0.2cm \parindent=3cm \fontseries{sb} {\fontsize{24}{10}\selectfont \@tipodocumento } \par}
    \end{adjustwidth}
    \parskip=0cm \noindent \textcolor{gris1}{\rule{\textwidth}{1pt}} \par 
    
    % Antetitulo
    \begin{adjustwidth}{4.0cm}{0.5cm}
        \parbox{\linewidth}{
            { \parskip=0.8cm  \fontseries{sb} \fontsize{12}{20} \selectfont \hyphenpenalty=10000 
                \begin{flushleft} 
                    \@antetitulo
                \end{flushleft} 
            } 
        }
    \end{adjustwidth}

    % Titulo
    \begin{adjustwidth}{4.0cm}{0.5cm}
        \parbox{\linewidth}{
            { \parskip=0.8cm \fontseries{l} \fontsize{24}{32} \selectfont \hyphenpenalty=10000 
                \begin{flushleft} 
                    \@title 
                \end{flushleft} 
            }
        }
    \end{adjustwidth}
    
    % Autor
    \begin{adjustwidth}{4.0cm}{0.5cm}
        \parbox{\linewidth}{
            \begin{flushleft} 
                { \parskip=1.8cm \fontseries{sb} \fontsize{12}{20} \selectfont \hyphenpenalty=10000 
                 \@calidadautor :} \\
                 { \parskip=0.3cm \fontseries{l} \fontsize{12}{20} \selectfont \hyphenpenalty=10000 
                 \@author }   
                
            \end{flushleft} 
        }
    \end{adjustwidth}
    
    % Fecha y lugar
     \begin{adjustwidth}{4.0cm}{5cm}
        \parbox{\linewidth}{
            {\parskip=2cm  \fontseries{l} \fontsize{12}{20} \selectfont \hyphenpenalty=10000 
                \begin{flushleft} 
                    \@address \  \textbar \  \@date
                \end{flushleft} 
            }
        }
    \end{adjustwidth}   
    
    
    \begin{adjustwidth}{3.0cm}{5cm}
        \vspace{0.4cm}
        \protect \noindent \includegraphics[align=t, height=1.6cm]{estadonacion/logoConare.png}
        \protect \noindent \includegraphics[align=t, height=1cm]{estadonacion/logo-pen-letraazul.png}
    \end{adjustwidth}
    
    \newpage
    \restoregeometry
    \setlength{\parskip}{0.8em}
    
}


% Plantilla para las figuras

\addto\captionsspanish{\renewcommand{\figurename}{Gráfico}}
\renewcommand{\figurename}{Gráfico}

\newcommand{\grafico}[3]{
    \begin{figure}[H]
        \captionsetup{type=figure}
        \caption{#3}
        \includegraphics[width=1\textwidth]{#1}
        \label{#2}
    \end{figure}
}


\renewcommand{\normalsize}{\fontsize{10}{15}\selectfont}
\setlength{\parindent}{0pt}
\setcounter{secnumdepth}{3}


