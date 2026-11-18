variable "project" {
  type    = string
  default = "terrasentinel"
}
variable "environment" {
  type    = string
  default = "staging"
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be staging or production"
  }
}
variable "region" {
  type    = string
  default = "eu-central-1"
}
variable "vpc_cidr" {
  type    = string
  default = "10.40.0.0/16"
}
variable "image_tag" {
  type        = string
  description = "Container image tag pushed to ECR (e.g. the git SHA)"
  default     = "latest"
}
variable "api_cpu" {
  type    = number
  default = 512
}
variable "api_memory" {
  type    = number
  default = 1024
}
variable "worker_cpu" {
  type    = number
  default = 2048
}
variable "worker_memory" {
  type        = number
  default     = 8192
  description = "Tiled inference keeps memory bounded, but models + GDAL caches need headroom"
}
variable "worker_desired_count" {
  type    = number
  default = 1
}
variable "db_instance_class" {
  type    = string
  default = "db.t4g.small"
}
variable "redis_node_type" {
  type    = string
  default = "cache.t4g.micro"
}
variable "alarm_email" {
  type    = string
  default = ""
}
variable "acm_certificate_arn" {
  type        = string
  default     = ""
  description = "If set, the ALB serves HTTPS and redirects HTTP"
}
