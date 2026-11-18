terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.70" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  # Configure a remote backend (S3 + DynamoDB lock) before any real apply:
  # backend "s3" { bucket = "...", key = "terrasentinel/terraform.tfstate", region = "...", dynamodb_table = "..." }
}

provider "aws" {
  region = var.region
  default_tags { tags = local.tags }
}
