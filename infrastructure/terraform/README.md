# Terraform — AWS deployment (not applied)

Validated locally with `terraform validate`; **no AWS resources have been created**. Applying this
costs money (NAT gateway, RDS, ElastiCache, Fargate, ALB) and requires an AWS account.

```
VPC (2 AZ) ── public: ALB ──► private: ECS Fargate api (×2) ─┬─► RDS PostgreSQL 16 (+PostGIS)
                                         ECS Fargate worker ─┼─► ElastiCache Redis (Celery)
                                                             └─► S3 (imagery, masks, models) via VPC endpoint
Secrets Manager (DB URL, app secret) · CloudWatch logs + metric filters + alarms → SNS
```

Before `apply`: configure the S3 remote backend in `versions.tf`, build & push ARM64 images to the
ECR repos (`api`, `worker` targets of `infrastructure/docker/backend.Dockerfile`), set
`image_tag`, optionally `acm_certificate_arn` and `alarm_email`. The GPU training workload is not
part of this stack (train on a GPU instance and register the model artifact in S3).

```bash
terraform init && terraform plan -var environment=staging
```
