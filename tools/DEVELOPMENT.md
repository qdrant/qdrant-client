# Client update checklist

For breaking changes:

* [ ] Create a new branch from the branch of upcoming release. 
  * E.g. `git checkout v0.7.0 && git pull && git checkout -b v0.7.0-my-changees`

For fixes:

* [ ] Create a new branch from master

---

* [ ] Create python virtual environment and install dependencies
  * `pip install virtualenv` - install venv manager
  * `virtualenv venv` - create virtual env
  * `source venv/bin/activate` - enter venv
  * `pip install poetry` - install package manager
  * `poetry install` - install all dependencies
* [ ] Generate REST code: `bash -x tools/generate_rest_client.sh` - will automatically fetch openapi from `qdrant:master`. 
* [ ] Generate gRPC code: `bash -x tools/generate_grpc_client.sh` - will automatically fetch proto from `qdrant:master`. 

---

* [ ] Check that tests are passing: see `tests/integration-tests.sh`
* [ ] Write new tests, if required
* [ ] Create PR into parent branch