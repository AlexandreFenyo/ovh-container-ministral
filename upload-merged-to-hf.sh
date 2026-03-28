#!/bin/bash
hf auth login --token "$hfkey"
hf repo create fenyo/ministral-8b-instruct-FAQ-MES-WEB --type model
hf upload fenyo/ministral-8b-instruct-FAQ-MES-WEB ./ministral-8b-instruct-merged .

