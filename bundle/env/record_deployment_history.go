package env

import "context"

// recordDeploymentHistoryVariable names the environment variable that opts the
// bundle into the deployment metadata service (DMS), which records deployment
// history and manages locking and resource state on the server.
const recordDeploymentHistoryVariable = "DATABRICKS_BUNDLE_RECORD_DEPLOYMENT_HISTORY"

// RecordDeploymentHistory returns the environment variable that opts the bundle
// into the deployment metadata service (DMS), which records deployment history
// and manages locking and resource state on the server.
func RecordDeploymentHistory(ctx context.Context) (string, bool) {
	return get(ctx, []string{
		recordDeploymentHistoryVariable,
	})
}
