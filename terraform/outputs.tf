output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "pgvector_host" {
  description = "pgvector service hostname"
  value       = "pgvector-postgresql.pgvector.svc.cluster.local"
}

output "qdrant_host" {
  description = "Qdrant service hostname"
  value       = "qdrant.qdrant.svc.cluster.local"
}

output "weaviate_host" {
  description = "Weaviate service hostname"
  value       = "weaviate.weaviate.svc.cluster.local"
}

output "grafana_url" {
  description = "Grafana URL"
  value       = "http://kube-prometheus-stack-grafana.monitoring.svc.cluster.local"
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${var.cluster_name}"
}
