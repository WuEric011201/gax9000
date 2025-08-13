# gax9000: Wafer Auto-Probe Measurement Controller

Controller for MIT novels group auto probe station. Manages measurements
with B1500A Semiconductor Parameter Analyzer and Cascade Summit 12000
auto-probe station.


# Repo structure
```
gax9000/
 ├─ controller/   - Python backend instrument controller
 └─ frontend/     - JS frontend controller and monitor
```

# Running ---------------CMU TONG WU
python --version
Python 3.10.11
pip --version
pip 23.0.1 
 .\venv\Scripts\activate
cd frontend; npm run dev 
New terminal run Backend: 	.\run.bat 
Open  https://localhost:80	
probe_gate = 4
probe_source = 5
probe_drain = 3


modules_file = "C:\\Users\\tongwu2\\Documents\\GitHub\\gax9000\\build\\measurements\\skywater_cnfet_sputter_scaling_v1_fet_modules.toml"

sweep_file = "C:\\Users\\tongwu2\\Documents\\GitHub\\gax9000\\build\\measurements\\final_width_sweep.toml"


# Setup/Installation(only for first time) 
1.  Create python virtual environment 
```
python -m venv venv
```

2.  Start environment
```
(Windows)
source .\venv\Scripts\activate
```

3.  Install python requirements
```
pip install -r requirements.txt
```

4.  Install controller backend as a pip package (required for scripts to find `controller` as dependency):
```
pip install -e .
```

5. Install frontend web ui packages
```
cd frontend
npm install
```

6. To Run the Frontend
cd .\frontend\
npm run dev

# Usage
First run server using script
```
./run.bat
```
This will by default use `./data` as the real instance data folder.


# Installation/Usage Issues
- If error occurs when generating SSL keys (for local HTTPS) on Windows, this may be
due to missing `OPENSSL_CONF` path. Open Windows > Environment Variables and add a
`OPENSSL_CONF`. If using miniconda, this can be set to:
```
OPENSSL_CONF: C:\Users\[USER]\miniconda3\Library\ssl\openssl.cnf
```

# Architecture
Python backend controller manages the instrument GPIB connections
and acts as a web server. The javascript frontend both acts to
update backend controller settings and run measurements, and 
reads real-time signals from controller web server (using SSE) and
reports measurement results.


# Issues
## Antivirus blocking SSE
Sophos antivirus (used by MIT) blocks SSE in python web apps from
regular http requests. It treats the SSE stream as a download that
needs to be scanned. It waits until a 2 MB block is downloaded before
scanning...issue is SSE events are small and unlikely to fill the 2 MB
buffer. There are 2 solutions
1.  Send a single 2 MB block before first event (to break through
    Sophos cancer).
2.  Use https (SSL/TLS)

https://stackoverflow.com/questions/62129788/on-a-machine-running-sophos-why-do-all-my-browsers-fail-to-receive-server-sent

## Generating self-signed SSL Certificate
See:
https://security.stackexchange.com/questions/74345/provide-subjectaltname-to-openssl-directly-on-the-command-line/198409#198409

The `run.sh` or `run.bat` script will generate a certificate file in 
`settings/ssl/cert.pem`. On windows, you will need to go into Certificates
and add `cert.pem` as a Trusted Root Certificate for local testing.

In Windows, go to
```
Manage User Certificates > Trusted Root Certification Authorities > Certificates
```
Add the generated `cert.pem` to the trusted certification authorities.

## SMU vs. Slot ID
The program `probe` settings are intended to use "SMU" inputs, similar
to how one would manually run programs on the B1500 tool. However,
over GPIB, we are actually sending `slot` numbers, NOT SMU numbers.
The B1500 slots are not necessarily configured to match SMU number.
e.g. Slot 2 could house SMU 3. In that case, the "probe" for SMU 3 should
actually be 2 not 3.

Optional config settings "smu_slots" maps SMU number to actual slot number,
which will need to be changed before sending data through GPIB.