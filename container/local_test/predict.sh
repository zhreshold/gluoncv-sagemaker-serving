#!/bin/bash

payload=$1
content=${2:-image/jpeg}

curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8080/invocations
