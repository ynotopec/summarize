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
#${pythonVersion} -m pip install -U -r requirements.txt
#optimize space
#jdupes -X size+:99M -r -L ~

export OPENAI_API_MODEL="vicuna"
export OPENAI_API_BASE="https://api-ai.ai-dev.numerique-interieur.com/v1"
export OPENAI_API_KEY="sk-a3OztiVj9cQyAK8ReQXAgw"

[ ! -z "${serverAddress}" ] && export GRADIO_SERVER_NAME="${serverAddress}"
[ ! -z "${portNumber}" ] && export GRADIO_SERVER_PORT="${portNumber}"
export CUDA_LAUNCH_BLOCKING=1

${pythonVersion} -m streamlit run app.py --browser.gatherUsageStats false $([ ! -z "${serverAddress}" ] && echo --server.address ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --server.port ${portNumber})
#${pythonVersion} app.py
#${pythonVersion} -m uvicorn app:app --reload $([ ! -z "${serverAddress}" ] && echo --host ${serverAddress}) $([ ! -z "${portNumber}" ] && echo --port ${portNumber})
