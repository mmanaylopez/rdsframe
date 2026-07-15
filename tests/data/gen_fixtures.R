args <- commandArgs(trailingOnly = TRUE)
out <- args[[1]]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

save <- function(object, name, ...) {
  saveRDS(object, file.path(out, paste0(name, ".rds")), ...)
}

# --- ALTREP candidates (only classes with Serialized_state survive saveRDS) ---
save(1:100000, "altrep_compact_intseq")
save(seq(2L, 200L, by = 2L), "altrep_intseq_step")
save(seq(1.5, 1000.5, by = 1), "altrep_compact_realseq")
save(as.character(1:5000), "altrep_deferred_string")
save(sort(c(30L, 10L, 20L, 40L)), "altrep_sorted_int_wrapper")
save(sort(c(3.5, 1.25, 2.75)), "altrep_sorted_real_wrapper")
x <- c("b", "a", "c")
save(sort(x), "altrep_sorted_string_wrapper")

# data.frame whose column is an ALTREP compact sequence
df_altrep <- data.frame(id = 1:50000)
df_altrep$value <- as.character(1:50000)
save(df_altrep, "df_with_altrep_columns")

# --- POSIXlt ---
lt <- as.POSIXlt(c("2024-03-15 10:30:45.5", NA), tz = "UTC")
save(lt, "posixlt_standalone")
df_lt <- data.frame(id = 1:2)
df_lt$when <- lt
save(df_lt, "df_posixlt_column")

# --- character row names ---
df_rn <- data.frame(x = c(10L, 20L, 30L), row.names = c("alpha", "beta", "gamma"))
save(df_rn, "df_character_rownames")

# --- default compact row names (control) ---
save(data.frame(x = 1:3), "df_default_rownames")

# --- difftime / ordered factor from real R ---
save(data.frame(d = as.difftime(c(1.5, 2.0), units = "hours")), "df_difftime_hours")
f <- factor(c("low", "high", "low"), levels = c("low", "medium", "high"), ordered = TRUE)
save(data.frame(f = f), "df_ordered_factor")

# --- native (non-XDR) format: saveRDS is serialize()-to-connection, so an
# uncompressed native-byte-order stream written to a file is a valid RDS ---
con <- file(file.path(out, "df_native_format.rds"), "wb")
serialize(data.frame(a = 1:3, b = c("x", "y", "z")), con, xdr = FALSE)
close(con)

# --- version 2 serialization ---
save(data.frame(a = 1:3), "df_version2", version = 2)

# --- deeply nested general object ---
nested <- list(
  meta = list(created = "2026-07-12", n = 42L),
  tables = list(data.frame(v = 1:2), data.frame(w = c("a", "b"))),
  matrix = matrix(1:6, nrow = 2)
)
save(nested, "nested_general_object")

cat("fixtures written to", out, "\n")
print(list.files(out))

# ---- second corpus: S4, environments, closures ----
args <- commandArgs(trailingOnly = TRUE)
out <- args[[1]]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

save <- function(object, name) {
  saveRDS(object, file.path(out, paste0(name, ".rds")))
}

# --- S4 object ---
setClass("Person", representation(name = "character", age = "numeric"))
save(new("Person", name = "Ana", age = 31), "s4_person")

# --- S4 with a nested data.frame slot ---
setClass("Study", representation(title = "character", data = "data.frame"))
save(new("Study", title = "catch survey", data = data.frame(x = 1:3)), "s4_with_dataframe")

# --- environment with values ---
e <- new.env()
e$alpha <- 1:3
e$beta <- "hello"
e$gamma <- data.frame(v = c(1.5, 2.5))
save(e, "environment_simple")

# --- environment referenced twice in a list (reference table alignment) ---
save(list(first = e, second = e), "environment_shared_ref")

# --- environment whose parent is a package namespace ---
ns_env <- new.env(parent = getNamespace("stats"))
ns_env$value <- 42L
save(ns_env, "environment_ns_parent")

# --- closure: must fail with a CLEAR error naming the type ---
save(function(x) x + 1, "closure_function")

# --- language call ---
save(quote(a + b), "language_call")

# --- formula (LANGSXP with class) ---
save(y ~ x + 1, "formula_object")

cat("fixtures2 written\n")
print(list.files(out))

