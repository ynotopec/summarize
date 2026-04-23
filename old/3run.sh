#!/bin/bash

serverAddress=$1
portNumber=$2

pythonVersion=python3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pythonDir=~/"venv/$(basename "${DIR}")"
cd $DIR

deactivate 2>/dev/null
mkdir -p "${pythonDir}"
${pythonVersion} -m venv "${pythonDir}"
source "${pythonDir}"/bin/activate

#intall
#${pythonVersion} -m pip cache purge
#${pythonVersion} -m pip install -U pip
#${pythonVersion} -m pip install -U pipreqs
#pipreqs .
#${pythonVersion} -m pip install -U -r requirements.txt
#jdupes -X size+:55M -r -L ~
#jdupes -r -L "${pythonDir}"

if [ -z "${OPENAI_API_KEY}" ] ;then
  #export OPENAI_API_BASE="https://api-vicuna.ai-dev.numerique-interieur.com/v1"
  #export OLLAMA_SERVER_URL="http://51.159.134.189:11434"
  #export OPENAI_API_BASE="https://api-mixtral.c1.ns1lab.net/v1"
  export OPENAI_API_BASE="https://api-vicuna.c0.ns1lab.net/v1"
  #export OPENAI_API_BASE="https://api-mistral.c0.ai-dev.numerique-interieur.com/v1"

  #unset OPENAI_API_BASE
  export OPENAI_API_KEY="EMPTY"
fi

${pythonVersion} -m streamlit run 3app.py --browser.gatherUsageStats false $([ ! -z "${serverAddress}" ] && echo --server.address ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --server.port ${portNumber})
#${pythonVersion} app.py
#${pythonVersion} -m uvicorn app:app --reload $([ ! -z "${serverAddress}" ] && echo --host ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --port ${portNumber})
