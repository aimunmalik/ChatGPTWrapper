terraform {
  backend "s3" {
    # Remaining configuration supplied at init time via:
    #   terraform init -backend-config=backend.hcl
    # See backend.hcl.example for the expected values.
  }
}
