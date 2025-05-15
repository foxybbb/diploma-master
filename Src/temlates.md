## Equation
```tex
\begin{ceqn} 
\begin{align} \label{kops}
    kOPS= \frac{1000}{T_{cal} - T_{io}}
 \end{align}
 \end{ceqn}
```


##  Citatation 

```tex
\citep{progit}
```

## Table

```tex
 \begin{table}[H]
    \caption{Title of the table}\label{tab:first}
    \centering
    \begin{tabular}{|l|l|l|l|l|}
    \hline
     &  &  &  &  \\ \hline
     &  &  &  &  \\ \hline
     &  &  &  &  \\ \hline
     &  &  &  &  \\ \hline
    \end{tabular}
 \end{table}
```

## Picture

```tex
\begin{figure}[H]
	\centering
	\includegraphics[width=0.5\textwidth]{PATH}
	\caption{CAPTION NAME}
	\label{123}
\end{figure}

------90deg
\begin{figure}[H]
    \begin{adjustbox}{addcode={\begin{minipage}{\width}}{\caption{
        Функциональная схема системы исполнительного управления
        } \label{FACD}
    \end{minipage}},rotate=90,center} 
        \includegraphics[width=0.85\paperheight]{Src/images/ACD (4).drawio.png}
    \end{adjustbox}
  \end{figure}
------svg

\begin{figure}[H]
	\centering
	\includesvg{Src/images/Hierarchy(1).svg}
	\caption{CAPTION NAME}
	\label{Hierar}
\end{figure}
```

## Items 

```tex
\begin{itemize}
	\item x1;
	\item x2;
	\item x3;
	\item x4.
\end{itemize}
```

## Items Center 

```tex
\begin{center}
	\begin{minipage}[c]{0.5\linewidth}
		\begin{itemize}[label={}]
			\item $ \mathbf{e} $ - euler's number 2.71828
			\item Item 2
			\item Item 3
		\end{itemize}
	\end{minipage}
\end{center}
```

## Code

```tex
\begin{figure}[H]
	\centering
	\begin{minted}[tabsize=2,breaklines,fontsize=\small]{cpp}
        ## CODE HERE
	\end{minted}
	\caption{Event subscribtion example}
\end{figure}

```


```
dwebp file.webp -o file.png
```
