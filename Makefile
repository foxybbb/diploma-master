all: build run


build:
	lualatex -synctex=1 -interaction=nonstopmode -jobname=diploma-master -file-line-error main.tex

run:
	docker run -t -d -v ${PWD}:/diploma-master:Z latex-docker

clean:
	rm *.aux \
	*.fdb_latexmk \
	*.fls \
	*.lof \
	*.lot \
	*.log \
	*.out \
	*.synctex.gz \
	*.xdv \
	*.toc \
	*.xwm \
	*.aux

docker: 
	docker build -t latex-docker .
	docker run --rm -ti -v ${PWD}:/diploma-master:Z latex-docker bash -c "make build && make clean"