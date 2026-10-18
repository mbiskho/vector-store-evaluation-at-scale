resource "helm_release" "pgvector" {
  name             = "pgvector"
  repository       = "https://charts.bitnami.com/bitnami"
  chart            = "postgresql"
  namespace        = "pgvector"
  create_namespace = true

  set {
    name  = "primary.persistence.size"
    value = "${var.pgvector_storage_gb}Gi"
  }

  set {
    name  = "primary.resources.requests.memory"
    value = "16Gi"
  }

  set {
    name  = "primary.resources.requests.cpu"
    value = "4"
  }

  set {
    name  = "auth.database"
    value = "vectordb"
  }

  set {
    name  = "auth.username"
    value = "postgres"
  }

  set {
    name  = "auth.password"
    value = "postgres"
  }
}

resource "helm_release" "qdrant" {
  name             = "qdrant"
  repository       = "https://qdrant.github.io/qdrant-helm"
  chart            = "qdrant"
  namespace        = "qdrant"
  create_namespace = true

  set {
    name  = "persistence.size"
    value = "${var.qdrant_storage_gb}Gi"
  }

  set {
    name  = "resources.requests.memory"
    value = "16Gi"
  }

  set {
    name  = "resources.requests.cpu"
    value = "4"
  }
}

resource "helm_release" "weaviate" {
  name             = "weaviate"
  repository       = "https://weaviate.github.io/weaviate-helm"
  chart            = "weaviate"
  namespace        = "weaviate"
  create_namespace = true

  set {
    name  = "storage.size"
    value = "${var.weaviate_storage_gb}Gi"
  }

  set {
    name  = "resources.requests.memory"
    value = "16Gi"
  }

  set {
    name  = "resources.requests.cpu"
    value = "4"
  }

  set {
    name  = "env.AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED"
    value = "true"
  }
}
