#!/usr/bin/env bash

jube run openblas-0.3.23-gcc12.3.xml
jube run openblas-0.3.27-gcc13.3.xml
jube run openblas-0.3.30-gcc14.3.xml

jube run imkl-2023.2.0-gcc12.3.xml
jube run imkl-2024.2.0-gcc13.3.xml
jube run imkl-2025.1.0-gcc14.3.xml

jube run aoclblas-5.1-gcc14.3.xml
