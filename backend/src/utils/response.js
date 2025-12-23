export function success(res, data = null, msg = "ok") {
  return res.status(200).json({ code: 0, msg, data });
}

export function error(res, msg = "error", httpStatus = 400) {
  return res.status(httpStatus).json({ code: 1, msg });
}
