variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "vsebench-cluster"
}

variable "instance_type" {
  description = "EC2 instance type for worker nodes"
  type        = string
  default     = "m6i.2xlarge"
}

variable "node_count" {
  description = "Number of worker nodes"
  type        = number
  default     = 3
}

variable "pgvector_storage_gb" {
  description = "EBS volume size for pgvector (GB)"
  type        = number
  default     = 200
}

variable "qdrant_storage_gb" {
  description = "EBS volume size for Qdrant (GB)"
  type        = number
  default     = 200
}

variable "weaviate_storage_gb" {
  description = "EBS volume size for Weaviate (GB)"
  type        = number
  default     = 200
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default = {
    Project = "vsebench"
    Purpose = "vector-store-benchmark"
  }
}
