const blogDateFormatter = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hourCycle: 'h23',
});

type BlogDateParts = {
  year: string;
  month: string;
  day: string;
  hour: string;
  minute: string;
  second: string;
};

export function getBlogDateParts(date: Date): BlogDateParts {
  const parts = Object.fromEntries(
    blogDateFormatter.formatToParts(date).map((part) => [part.type, part.value])
  );

  return {
    year: parts.year,
    month: parts.month,
    day: parts.day,
    hour: parts.hour,
    minute: parts.minute,
    second: parts.second,
  };
}

export function formatBlogDateTime(date: Date): string {
  const { year, month, day, hour, minute, second } = getBlogDateParts(date);
  return `${year}/${month}/${day} ${hour}:${minute}:${second}`;
}
