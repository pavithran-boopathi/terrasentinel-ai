output "alb_dns_name" { value = aws_lb.main.dns_name }
output "data_bucket" { value = aws_s3_bucket.data.bucket }
output "ecr_repositories" { value = { for k, r in aws_ecr_repository.app : k => r.repository_url } }
output "db_endpoint" { value = aws_db_instance.main.address }
output "app_secret_arn" { value = aws_secretsmanager_secret.app.arn }
