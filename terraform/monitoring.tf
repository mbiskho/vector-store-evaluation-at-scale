resource "helm_release" "prometheus_stack" {
  name             = "kube-prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true

  set {
    name  = "grafana.adminPassword"
    value = "admin"
  }

  set {
    name  = "prometheus.server.retention"
    value = "7d"
  }
}

resource "kubernetes_config_map" "grafana_dashboards" {
  metadata {
    name      = "vsebench-dashboards"
    namespace = "monitoring"
    labels = {
      grafana_dashboard = "1"
    }
  }

  data = {
    "vector-store-benchmark.json" = file("${path.module}/../docker/grafana/dashboards/benchmark.json")
  }
}
