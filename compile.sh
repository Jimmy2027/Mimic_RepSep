#!/usr/bin/env bash

TARGET="${1}"
WHITELIST="

midl_2021/slides_midl2021.tex

"
#midl_2021/midl-shortpaper.tex
#poster/midl2021_poster.tex
#  article.tex



if [[ "$TARGET" = "all" ]] || [[ "$TARGET" == "" ]]; then
	for ITER_TARGET in $WHITELIST; do

    ITER_TARGET=${ITER_TARGET%".tex"}
    ./compile.sh "${ITER_TARGET}"

	done

else
  echo "${TARGET}"
  if [ "${TARGET}" = "poster/midl2021_poster" ]; then
#    the poster need to be run with lualatex.
    lualatex -shell-escape "${TARGET}.tex" || { echo "Initial lualetex failed"; exit $ERRCODE; }
    bibtex "$(basename "${TARGET}")" || { echo "bibtex failed"; exit $ERRCODE; }
    lualatex -shell-escape "${TARGET}.tex" || { echo "Second lualetex failed"; exit $ERRCODE; }

  else
    pdflatex -shell-escape "${TARGET}.tex" || { echo "Initial pdflatex failed"; exit $ERRCODE; }

    #  Only execute pythontex if indicated in latex file
    first_line=$(head -n 1 "${TARGET}.tex")
    if [ "${first_line}" = "% pythontex" ]; then
        pythontex "${TARGET}.tex" || { echo "PythonTeX failed"; exit $ERRCODE; }
        pdflatex -shell-escape "${TARGET}.tex" || { echo "pdflatex failed after PythonTeX"; exit $ERRCODE; }
    fi

    bibtex "$(basename "${TARGET}")" || { echo "bibtex failed"; exit $ERRCODE; }
    pdflatex -shell-escape "${TARGET}.tex" || { echo "pdflatex failed after bibtex"; exit $ERRCODE; }
    pdflatex -shell-escape "${TARGET}.tex"

  fi

  # move file to zotero
  python upload.py "$(pwd)/$(basename "${TARGET}.pdf")"

  # cleanup latex logs
  if [ ! -d tex_logs ]; then
      mkdir tex_logs
  fi
	for CLEAN_TARGET in "*.aux *.log *.out *.bbl *.pytxcode *blx.bib *.blg *.run.xml *.bcf *.nav *.toc *.snm"; do
	  mv $CLEAN_TARGET tex_logs/
  done


fi

