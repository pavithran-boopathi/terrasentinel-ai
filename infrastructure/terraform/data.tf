resource "random_password" "db" {
  length  = 32
  special = false
}

resource "random_password" "secret_key" {
  length  = 64
  special = false
}

resource "aws_secretsmanager_secret" "app" {
  name_prefix             = "${local.name}-app-"
  recovery_window_in_days = var.environment == "production" ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode({
    TS_SECRET_KEY   = random_password.secret_key.result
    TS_DATABASE_URL = "postgresql+psycopg://terrasentinel:${random_password.db.result}@${aws_db_instance.main.address}:5432/terrasentinel"
  })
}

resource "aws_db_subnet_group" "main" {
  name       = local.name
  subnet_ids = aws_subnet.private[*].id
}

# PostGIS is enabled by the first Alembic migration (CREATE EXTENSION postgis), supported on RDS.
resource "aws_db_instance" "main" {
  identifier                   = local.name
  engine                       = "postgres"
  engine_version               = "16"
  instance_class               = var.db_instance_class
  allocated_storage            = 20
  max_allocated_storage        = 200
  storage_encrypted            = true
  db_name                      = "terrasentinel"
  username                     = "terrasentinel"
  password                     = random_password.db.result
  db_subnet_group_name         = aws_db_subnet_group.main.name
  vpc_security_group_ids       = [aws_security_group.data.id]
  publicly_accessible          = false
  backup_retention_period      = var.environment == "production" ? 14 : 1
  deletion_protection          = var.environment == "production"
  skip_final_snapshot          = var.environment != "production"
  final_snapshot_identifier    = var.environment == "production" ? "${local.name}-final" : null
  performance_insights_enabled = true
  auto_minor_version_upgrade   = true
}

resource "aws_elasticache_subnet_group" "main" {
  name       = local.name
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id         = local.name
  engine             = "redis"
  node_type          = var.redis_node_type
  num_cache_nodes    = 1
  port               = 6379
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.data.id]
}
