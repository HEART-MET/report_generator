filename=root

pdf:
	mkdir -p build
	pdflatex --interaction=batchmode --output-directory build ${filename}
	mv build/${filename}.pdf .

read:
	evince build/${filename}.pdf &

clean:
	rm -f build/${filename}.{ps,pdf,log,aux,out,dvi,bbl,blg}

