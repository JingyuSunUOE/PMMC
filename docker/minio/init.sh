#!/bin/sh
set -eu

validate_bucket_name() {
  case "$1" in
    ""|*[!a-z0-9.-]*)
      printf 'Invalid S3 bucket name: only lowercase letters, digits, dots, and hyphens are supported.\n' >&2
      exit 1
      ;;
  esac
}

write_runtime_policy() {
  printf '%s\n' \
    '{' \
    '  "Version": "2012-10-17",' \
    '  "Statement": [' \
    '    {' \
    '      "Effect": "Allow",' \
    '      "Action": ["s3:GetBucketLocation", "s3:ListBucket"],' \
    "      \"Resource\": [\"arn:aws:s3:::$PMMC_OBJECT_STORE_BUCKET\"]" \
    '    },' \
    '    {' \
    '      "Effect": "Allow",' \
    '      "Action": ["s3:GetObject", "s3:PutObject"],' \
    "      \"Resource\": [\"arn:aws:s3:::$PMMC_OBJECT_STORE_BUCKET/*\"]" \
    '    }' \
    '  ]' \
    '}'
}

write_evaluator_policy() {
  printf '%s\n' \
    '{' \
    '  "Version": "2012-10-17",' \
    '  "Statement": [' \
    '    {' \
    '      "Effect": "Allow",' \
    '      "Action": ["s3:GetBucketLocation", "s3:ListBucket"],' \
    '      "Resource": [' \
    "        \"arn:aws:s3:::$PMMC_OBJECT_STORE_BUCKET\"," \
    "        \"arn:aws:s3:::$PMMC_OBJECT_STORE_EVALUATION_BUCKET\"" \
    '      ]' \
    '    },' \
    '    {' \
    '      "Effect": "Allow",' \
    '      "Action": ["s3:GetObject"],' \
    "      \"Resource\": [\"arn:aws:s3:::$PMMC_OBJECT_STORE_BUCKET/*\"]" \
    '    },' \
    '    {' \
    '      "Effect": "Allow",' \
    '      "Action": ["s3:GetObject", "s3:PutObject"],' \
    "      \"Resource\": [\"arn:aws:s3:::$PMMC_OBJECT_STORE_EVALUATION_BUCKET/*\"]" \
    '    }' \
    '  ]' \
    '}'
}

validate_bucket_name "$PMMC_OBJECT_STORE_BUCKET"
validate_bucket_name "$PMMC_OBJECT_STORE_EVALUATION_BUCKET"

policy_output_dir="${PMMC_POLICY_OUTPUT_DIR:-/tmp}"
runtime_policy_path="$policy_output_dir/runtime-policy.json"
evaluator_policy_path="$policy_output_dir/evaluator-policy.json"
minio_endpoint="${PMMC_MINIO_ENDPOINT:-http://minio:9000}"

mc alias set local "$minio_endpoint" "$PMMC_MINIO_ROOT_USER" "$PMMC_MINIO_ROOT_PASSWORD"
mc mb --ignore-existing "local/$PMMC_OBJECT_STORE_BUCKET"
mc mb --ignore-existing "local/$PMMC_OBJECT_STORE_EVALUATION_BUCKET"

write_runtime_policy > "$runtime_policy_path"
write_evaluator_policy > "$evaluator_policy_path"

mc admin user add local "$PMMC_OBJECT_STORE_ACCESS_KEY" "$PMMC_OBJECT_STORE_SECRET_KEY" || true
mc admin user add local \
  "$PMMC_OBJECT_STORE_EVALUATOR_ACCESS_KEY" \
  "$PMMC_OBJECT_STORE_EVALUATOR_SECRET_KEY" || true
mc admin policy create local pmmc-runtime "$runtime_policy_path"
mc admin policy create local pmmc-evaluator "$evaluator_policy_path"
mc admin policy attach local pmmc-runtime --user "$PMMC_OBJECT_STORE_ACCESS_KEY"
mc admin policy attach local pmmc-evaluator --user "$PMMC_OBJECT_STORE_EVALUATOR_ACCESS_KEY"
