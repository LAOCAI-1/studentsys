export function errorHandler(err, _req, res, _next) {
  console.error("[unhandled error]", err);
  res.status(500).json({ code: 1, msg: "服务器错误" });
}
