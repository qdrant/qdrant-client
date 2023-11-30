# Client update checklist

For breaking changes:

* [ ] Create a new branch from the branch of upcoming release.
  * E.g. `git checkout v0.7.0 && git pull && git checkout -b v0.7.0-my-changees`

For fixes:

* [ ] Create a new branch from master

---

* [ ] Create python virtual environment and install dependencies
  * Install pyenv https://github.com/pyenv/pyenv#automatic-installer
  * Install system pyenv dependencies https://github.com/pyenv/pyenv/wiki#suggested-build-environment
  * `pyenv install 3.10.10` - install python 3.10.10
  * `pyenv local 3.10.10` - set python version
  * `pip install grpcio==1.59.3` - install grpcio
  * `pip install grpcio-tools==1.59.3` - install grpcio-tools
  * `pip install virtualenv` - install venv manager
  * `virtualenv venv` - create virtual env
  * `source venv/bin/activate` - enter venv
  * `pip install poetry` - install package manager
  * `poetry install` - install all dependencies
* [ ] (MacOS) make sure to have `gnu-sed` installed and aliased to `sed`: `brew install gnu-sed`. Instructions to set alias can be found via `brew info gnu-sed`
* [ ] Generate REST code: `bash -x tools/generate_rest_client.sh` - will automatically fetch openapi from `qdrant:dev`.
* [ ] Generate gRPC code: `bash -x tools/generate_grpc_client.sh` - will automatically fetch proto from `qdrant:dev`.

---

* [ ] Check that tests are passing: see `tests/integration-tests.sh`
* [ ] Write new tests, if required
* [ ] Create PR into parent branch
