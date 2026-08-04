/**
 * cnumpy datetime support.
 *
 * Public scalar values are NumPy datetime64/timedelta64 int64 payloads in the
 * explicitly supplied unit.  Fine-unit construction intentionally performs
 * two's-complement modulo arithmetic, matching NumPy's int64 payload rules.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define CNP_DATETIME_NAT INT64_MIN
#define CNP_SECONDS_PER_DAY INT64_C(86400)

static const char *const g_datetime_unit_names[] = {
    "Y", "M", "W", "D", "h", "m", "s",
    "ms", "us", "ns", "ps", "fs", "as"
};

static bool datetime_unit_valid(CNP_DATETIME_UNIT unit) {
    return unit >= CNP_FR_Y && unit <= CNP_FR_as;
}

static bool require_datetime_unit(
    CNP_DATETIME_UNIT unit, const char *function_name) {
    if (datetime_unit_valid(unit)) return true;
    cnp_set_error(
        CNP_ERR_VALUE, function_name,
        "datetime unit %d is outside the supported range", (int)unit);
    return false;
}

static int64_t floor_div_i64(int64_t value, int64_t divisor) {
    int64_t quotient = value / divisor;
    int64_t remainder = value % divisor;
    if (remainder < 0) --quotient;
    return quotient;
}

static int64_t floor_mod_i64(int64_t value, int64_t divisor) {
    int64_t remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}

static bool checked_add_i64(int64_t left, int64_t right, int64_t *result) {
    if ((right > 0 && left > INT64_MAX - right) ||
        (right < 0 && left < INT64_MIN - right)) return false;
    *result = left + right;
    return true;
}

static bool checked_multiply_i64(
    int64_t left, int64_t right, int64_t *result) {
    if (left == 0 || right == 0) {
        *result = 0;
        return true;
    }
    if (left == -1) {
        if (right == INT64_MIN) return false;
        *result = -right;
        return true;
    }
    if (right == -1) {
        if (left == INT64_MIN) return false;
        *result = -left;
        return true;
    }
    if (left > 0) {
        if (right > 0) {
            if (left > INT64_MAX / right) return false;
        } else if (right < INT64_MIN / left) {
            return false;
        }
    } else if (right > 0) {
        if (left < INT64_MIN / right) return false;
    } else if (left < INT64_MAX / right) {
        return false;
    }
    *result = left * right;
    return true;
}

static int days_in_month(int64_t year, int month) {
    static const int lengths[] = {
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    };
    bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    return month == 2 && leap ? 29 : lengths[month - 1];
}

static bool valid_date(int64_t year, int month, int day) {
    (void)year;
    return month >= 1 && month <= 12 &&
        day >= 1 && day <= days_in_month(year, month);
}

/* Howard Hinnant's proleptic-Gregorian algorithm, with checked era math. */
static bool days_from_civil(
    int64_t year, int month, int day, int64_t *result) {
    if (!valid_date(year, month, day)) return false;
    int64_t adjusted_year = year;
    if (month <= 2) {
        if (adjusted_year == INT64_MIN) return false;
        --adjusted_year;
    }
    int64_t era = floor_div_i64(adjusted_year, 400);
    int64_t era_days;
    if (!checked_multiply_i64(era, 146097, &era_days)) return false;
    unsigned year_of_era = (unsigned)(adjusted_year - era * 400);
    unsigned shifted_month = (unsigned)(month + (month > 2 ? -3 : 9));
    unsigned day_of_year = (153 * shifted_month + 2) / 5 +
        (unsigned)day - 1;
    unsigned day_of_era = year_of_era * 365 + year_of_era / 4 -
        year_of_era / 100 + day_of_year;
    int64_t absolute_days;
    if (!checked_add_i64(era_days, (int64_t)day_of_era, &absolute_days) ||
        !checked_add_i64(absolute_days, -719468, result)) return false;
    return true;
}

static bool civil_from_days(
    int64_t days, int64_t *year, int *month, int *day) {
    int64_t era = floor_div_i64(days, 146097);
    int64_t day_of_era_signed = days - era * 146097;
    int64_t adjusted = day_of_era_signed + 135080;
    if (adjusted >= 146097) {
        adjusted -= 146097;
        if (era == INT64_MAX) return false;
        ++era;
    }
    if (!checked_add_i64(era, 4, &era)) return false;
    unsigned day_of_era = (unsigned)adjusted;
    unsigned year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 -
         day_of_era / 146096) / 365;
    int64_t era_years;
    if (!checked_multiply_i64(era, 400, &era_years) ||
        !checked_add_i64(era_years, (int64_t)year_of_era, year)) return false;
    unsigned day_of_year = day_of_era -
        (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    unsigned month_piece = (5 * day_of_year + 2) / 153;
    *day = (int)(day_of_year - (153 * month_piece + 2) / 5 + 1);
    *month = (int)(month_piece + (month_piece < 10 ? 3 : -9));
    if (*month <= 2) {
        if (*year == INT64_MAX) return false;
        ++*year;
    }
    return true;
}

static uint64_t power_of_ten(int digits) {
    uint64_t result = 1;
    for (int index = 0; index < digits; ++index) result *= UINT64_C(10);
    return result;
}

static int datetime_fraction_digits(CNP_DATETIME_UNIT unit) {
    switch (unit) {
        case CNP_FR_ms: return 3;
        case CNP_FR_us: return 6;
        case CNP_FR_ns: return 9;
        case CNP_FR_ps: return 12;
        case CNP_FR_fs: return 15;
        case CNP_FR_as: return 18;
        default: return 0;
    }
}

static int64_t datetime_ticks_per_second(CNP_DATETIME_UNIT unit) {
    return (int64_t)power_of_ten(datetime_fraction_digits(unit));
}

static int64_t int64_from_bits(uint64_t bits) {
    int64_t result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static bool components_to_ticks(
    int64_t year, int month, int day,
    int hour, int minute, int second,
    uint64_t fraction, int fraction_digits,
    CNP_DATETIME_UNIT unit, int64_t *result) {
    if (!valid_date(year, month, day) ||
        hour < 0 || hour > 23 || minute < 0 || minute > 59 ||
        second < 0 || second > 59) return false;
    if (unit == CNP_FR_Y) return checked_add_i64(year, -1970, result);
    if (unit == CNP_FR_M) {
        int64_t year_offset;
        if (!checked_add_i64(year, -1970, &year_offset) ||
            !checked_multiply_i64(year_offset, 12, &year_offset) ||
            !checked_add_i64(year_offset, month - 1, result)) return false;
        return true;
    }

    int64_t days;
    if (!days_from_civil(year, month, day, &days)) return false;
    if (unit == CNP_FR_W) {
        *result = floor_div_i64(days, 7);
        return true;
    }
    if (unit == CNP_FR_D) {
        *result = days;
        return true;
    }

    uint64_t ticks = (uint64_t)days;
    if (unit == CNP_FR_h) {
        ticks = ticks * UINT64_C(24) + (uint64_t)hour;
    } else if (unit == CNP_FR_m) {
        ticks = ticks * UINT64_C(1440) +
            (uint64_t)(hour * 60 + minute);
    } else {
        uint64_t per_second = (uint64_t)datetime_ticks_per_second(unit);
        uint64_t seconds = (uint64_t)(hour * 3600 + minute * 60 + second);
        ticks = (ticks * UINT64_C(86400) + seconds) * per_second;
        int target_digits = datetime_fraction_digits(unit);
        uint64_t fraction_ticks = fraction;
        if (fraction_digits > target_digits) {
            fraction_ticks /= power_of_ten(fraction_digits - target_digits);
        } else if (fraction_digits < target_digits) {
            fraction_ticks *= power_of_ten(target_digits - fraction_digits);
        }
        ticks += fraction_ticks;
    }
    *result = int64_from_bits(ticks);
    return true;
}

static bool ticks_to_components(
    int64_t value, CNP_DATETIME_UNIT unit,
    int64_t *year, int *month, int *day,
    int *hour, int *minute, int *second,
    uint64_t *fraction, int *fraction_digits) {
    if (value == CNP_DATETIME_NAT || !datetime_unit_valid(unit)) return false;
    *hour = 0;
    *minute = 0;
    *second = 0;
    *fraction = 0;
    *fraction_digits = datetime_fraction_digits(unit);

    if (unit == CNP_FR_Y) {
        if (!checked_add_i64(value, 1970, year)) return false;
        *month = 1;
        *day = 1;
        return true;
    }
    if (unit == CNP_FR_M) {
        int64_t year_offset = floor_div_i64(value, 12);
        if (!checked_add_i64(year_offset, 1970, year)) return false;
        *month = (int)floor_mod_i64(value, 12) + 1;
        *day = 1;
        return true;
    }

    int64_t days;
    int64_t seconds_of_day = 0;
    if (unit == CNP_FR_W) {
        if (!checked_multiply_i64(value, 7, &days)) return false;
    } else if (unit == CNP_FR_D) {
        days = value;
    } else if (unit == CNP_FR_h) {
        days = floor_div_i64(value, 24);
        *hour = (int)floor_mod_i64(value, 24);
    } else if (unit == CNP_FR_m) {
        days = floor_div_i64(value, 1440);
        int64_t minute_of_day = floor_mod_i64(value, 1440);
        *hour = (int)(minute_of_day / 60);
        *minute = (int)(minute_of_day % 60);
    } else {
        int64_t per_second = datetime_ticks_per_second(unit);
        int64_t total_seconds = floor_div_i64(value, per_second);
        *fraction = (uint64_t)floor_mod_i64(value, per_second);
        days = floor_div_i64(total_seconds, CNP_SECONDS_PER_DAY);
        seconds_of_day = floor_mod_i64(total_seconds, CNP_SECONDS_PER_DAY);
        *hour = (int)(seconds_of_day / 3600);
        *minute = (int)((seconds_of_day % 3600) / 60);
        *second = (int)(seconds_of_day % 60);
    }
    return civil_from_days(days, year, month, day);
}

static bool format_datetime(
    int64_t value, CNP_DATETIME_UNIT unit,
    char *buffer, size_t capacity, const char *function_name) {
    if (!require_datetime_unit(unit, function_name)) return false;
    if (value == CNP_DATETIME_NAT) {
        if (capacity < 4) {
            cnp_set_error(CNP_ERR_VALUE, function_name, "output buffer is too small");
            return false;
        }
        memcpy(buffer, "NaT", 4);
        return true;
    }
    int64_t year;
    int month, day, hour, minute, second, digits;
    uint64_t fraction;
    if (!ticks_to_components(
            value, unit, &year, &month, &day,
            &hour, &minute, &second, &fraction, &digits)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "datetime payload cannot be represented in this unit");
        return false;
    }

    int written;
    if (unit == CNP_FR_Y) {
        written = snprintf(buffer, capacity, "%04lld", (long long)year);
    } else if (unit == CNP_FR_M) {
        written = snprintf(
            buffer, capacity, "%04lld-%02d", (long long)year, month);
    } else if (unit <= CNP_FR_D) {
        written = snprintf(
            buffer, capacity, "%04lld-%02d-%02d",
            (long long)year, month, day);
    } else if (unit == CNP_FR_h) {
        written = snprintf(
            buffer, capacity, "%04lld-%02d-%02dT%02d",
            (long long)year, month, day, hour);
    } else if (unit == CNP_FR_m) {
        written = snprintf(
            buffer, capacity, "%04lld-%02d-%02dT%02d:%02d",
            (long long)year, month, day, hour, minute);
    } else if (unit == CNP_FR_s) {
        written = snprintf(
            buffer, capacity, "%04lld-%02d-%02dT%02d:%02d:%02d",
            (long long)year, month, day, hour, minute, second);
    } else {
        written = snprintf(
            buffer, capacity,
            "%04lld-%02d-%02dT%02d:%02d:%02d.%0*llu",
            (long long)year, month, day, hour, minute, second,
            digits, (unsigned long long)fraction);
    }
    if (written < 0 || (size_t)written >= capacity) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "output buffer is too small");
        return false;
    }
    return true;
}

static bool parse_int64_component(
    const char **cursor, bool signed_value, int64_t *result) {
    const char *position = *cursor;
    bool negative = false;
    if (signed_value && (*position == '+' || *position == '-')) {
        negative = *position == '-';
        ++position;
    }
    if (*position < '0' || *position > '9') return false;
    uint64_t limit = negative
        ? (uint64_t)INT64_MAX + UINT64_C(1) : (uint64_t)INT64_MAX;
    uint64_t value = 0;
    while (*position >= '0' && *position <= '9') {
        unsigned digit = (unsigned)(*position - '0');
        if (value > (limit - digit) / 10) return false;
        value = value * 10 + digit;
        ++position;
    }
    if (negative && value == (uint64_t)INT64_MAX + UINT64_C(1)) {
        *result = INT64_MIN;
    } else {
        *result = negative ? -(int64_t)value : (int64_t)value;
    }
    *cursor = position;
    return true;
}

static bool parse_iso_datetime(
    const char *text, CNP_DATETIME_UNIT unit, int64_t *result) {
    if (!text) return false;
    if (strcmp(text, "NaT") == 0) {
        *result = CNP_DATETIME_NAT;
        return true;
    }
    const char *cursor = text;
    int64_t year, component;
    int month = 1, day = 1, hour = 0, minute = 0, second = 0;
    uint64_t fraction = 0;
    int fraction_digits = 0;
    if (!parse_int64_component(&cursor, true, &year)) return false;
    if (*cursor == '-') {
        ++cursor;
        if (!parse_int64_component(&cursor, false, &component) ||
            component > INT_MAX) return false;
        month = (int)component;
        if (*cursor == '-') {
            ++cursor;
            if (!parse_int64_component(&cursor, false, &component) ||
                component > INT_MAX) return false;
            day = (int)component;
        }
    }
    if (*cursor == 'T') {
        ++cursor;
        if (!parse_int64_component(&cursor, false, &component) ||
            component > INT_MAX) return false;
        hour = (int)component;
        if (*cursor == ':') {
            ++cursor;
            if (!parse_int64_component(&cursor, false, &component) ||
                component > INT_MAX) return false;
            minute = (int)component;
            if (*cursor == ':') {
                ++cursor;
                if (!parse_int64_component(&cursor, false, &component) ||
                    component > INT_MAX) return false;
                second = (int)component;
                if (*cursor == '.') {
                    ++cursor;
                    while (*cursor >= '0' && *cursor <= '9') {
                        if (fraction_digits == 18) return false;
                        fraction = fraction * 10 + (unsigned)(*cursor - '0');
                        ++fraction_digits;
                        ++cursor;
                    }
                    if (fraction_digits == 0) return false;
                }
            }
        }
    }
    if (*cursor != '\0') return false;
    return components_to_ticks(
        year, month, day, hour, minute, second,
        fraction, fraction_digits, unit, result);
}

CNP_API int64_t CNP_CALL cnp_datetime64_from_date(
    int64_t year, int month, int day, CNP_DATETIME_UNIT unit) {
    const char *function_name = "cnp_datetime64_from_date";
    int64_t result;
    if (!require_datetime_unit(unit, function_name)) return 0;
    if (!components_to_ticks(
            year, month, day, 0, 0, 0, 0, 0, unit, &result)) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "date components are invalid");
        return 0;
    }
    return result;
}

CNP_API int64_t CNP_CALL cnp_datetime64_from_time(
    int64_t year, int month, int day,
    int hour, int minute, int second, CNP_DATETIME_UNIT unit) {
    const char *function_name = "cnp_datetime64_from_time";
    int64_t result;
    if (!require_datetime_unit(unit, function_name)) return 0;
    if (!components_to_ticks(
            year, month, day, hour, minute, second,
            0, 0, unit, &result)) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "date or time components are invalid");
        return 0;
    }
    return result;
}

CNP_API int64_t CNP_CALL cnp_datetime64_from_string(
    const char *str, CNP_DATETIME_UNIT unit) {
    const char *function_name = "cnp_datetime64_from_string";
    int64_t result;
    if (!require_datetime_unit(unit, function_name)) return 0;
    if (!parse_iso_datetime(str, unit, &result)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "input must be a fully consumed ISO datetime or NaT");
        return 0;
    }
    return result;
}

CNP_API int64_t CNP_CALL cnp_datetime64_now(CNP_DATETIME_UNIT unit) {
    const char *function_name = "cnp_datetime64_now";
    if (!require_datetime_unit(unit, function_name)) return 0;
    time_t current = time(NULL);
    if (current == (time_t)-1) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "system UTC time is unavailable");
        return 0;
    }
    struct tm utc;
    if (gmtime_s(&utc, &current) != 0) {
        cnp_set_error(CNP_ERR_GENERIC, function_name, "cannot convert system UTC time");
        return 0;
    }
    int64_t result;
    if (!components_to_ticks(
            utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday,
            utc.tm_hour, utc.tm_min, utc.tm_sec,
            0, 0, unit, &result)) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "system time is out of range");
        return 0;
    }
    return result;
}

CNP_API void CNP_CALL cnp_datetime64_to_date(
    int64_t dt, CNP_DATETIME_UNIT unit,
    int64_t *year, int *month, int *day) {
    const char *function_name = "cnp_datetime64_to_date";
    if (!year || !month || !day) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "all date outputs are required");
        return;
    }
    if (!require_datetime_unit(unit, function_name)) return;
    int64_t next_year;
    int next_month, next_day, hour, minute, second, digits;
    uint64_t fraction;
    if (!ticks_to_components(
            dt, unit, &next_year, &next_month, &next_day,
            &hour, &minute, &second, &fraction, &digits)) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "datetime payload is NaT or out of range");
        return;
    }
    *year = next_year;
    *month = next_month;
    *day = next_day;
}

CNP_API void CNP_CALL cnp_datetime64_to_time(
    int64_t dt, CNP_DATETIME_UNIT unit,
    int *hour, int *minute, int *second) {
    const char *function_name = "cnp_datetime64_to_time";
    if (!hour || !minute || !second) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "all time outputs are required");
        return;
    }
    if (!require_datetime_unit(unit, function_name)) return;
    int64_t year;
    int month, day, next_hour, next_minute, next_second, digits;
    uint64_t fraction;
    if (!ticks_to_components(
            dt, unit, &year, &month, &day,
            &next_hour, &next_minute, &next_second, &fraction, &digits)) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "datetime payload is NaT or out of range");
        return;
    }
    *hour = next_hour;
    *minute = next_minute;
    *second = next_second;
}

CNP_API char* CNP_CALL cnp_datetime64_to_string(
    int64_t dt, CNP_DATETIME_UNIT unit) {
    const char *function_name = "cnp_datetime64_to_string";
    char temporary[96];
    if (!format_datetime(dt, unit, temporary, sizeof(temporary), function_name))
        return NULL;
    size_t length = strlen(temporary);
    char *result = (char*)cnp_malloc(length + 1);
    if (!result) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "cannot allocate datetime string");
        return NULL;
    }
    memcpy(result, temporary, length + 1);
    return result;
}

CNP_API int64_t CNP_CALL cnp_timedelta64_create(
    int64_t value, CNP_DATETIME_UNIT unit) {
    if (!require_datetime_unit(unit, "cnp_timedelta64_create")) return 0;
    return value;
}

CNP_API int64_t CNP_CALL cnp_datetime64_add(
    int64_t dt, int64_t delta, CNP_DATETIME_UNIT unit) {
    if (!require_datetime_unit(unit, "cnp_datetime64_add")) return 0;
    if (dt == CNP_DATETIME_NAT || delta == CNP_DATETIME_NAT)
        return CNP_DATETIME_NAT;
    return int64_from_bits((uint64_t)dt + (uint64_t)delta);
}

CNP_API int64_t CNP_CALL cnp_datetime64_subtract(
    int64_t dt1, int64_t dt2, CNP_DATETIME_UNIT unit) {
    if (!require_datetime_unit(unit, "cnp_datetime64_subtract")) return 0;
    if (dt1 == CNP_DATETIME_NAT || dt2 == CNP_DATETIME_NAT)
        return CNP_DATETIME_NAT;
    return int64_from_bits((uint64_t)dt1 - (uint64_t)dt2);
}

CNP_API int CNP_CALL cnp_datetime64_compare(int64_t dt1, int64_t dt2) {
    if (dt1 == CNP_DATETIME_NAT || dt2 == CNP_DATETIME_NAT) {
        cnp_set_error(
            CNP_ERR_VALUE, "cnp_datetime64_compare",
            "NaT is unordered and cannot be represented by this comparison ABI");
        return 0;
    }
    return dt1 < dt2 ? -1 : dt1 > dt2 ? 1 : 0;
}

static int weekday_from_days(int64_t days) {
    return (int)((floor_mod_i64(days, 7) + 3) % 7);
}

static bool is_business_day_raw(int64_t days) {
    return days != CNP_DATETIME_NAT && weekday_from_days(days) < 5;
}

CNP_API bool CNP_CALL cnp_is_busday(int64_t dt) {
    return is_business_day_raw(dt);
}

static bool busday_count_forward(
    int64_t start, int64_t end, int64_t *result) {
    uint64_t span = (uint64_t)end - (uint64_t)start;
    uint64_t weeks = span / 7;
    if (weeks > (uint64_t)INT64_MAX / 5) return false;
    int64_t count = (int64_t)(weeks * 5);
    uint64_t day_bits = (uint64_t)start + weeks * 7;
    int64_t day = int64_from_bits(day_bits);
    unsigned remainder = (unsigned)(span % 7);
    for (unsigned index = 0; index < remainder; ++index) {
        if (is_business_day_raw(day)) ++count;
        day = int64_from_bits((uint64_t)day + UINT64_C(1));
    }
    *result = count;
    return true;
}

CNP_API int64_t CNP_CALL cnp_busday_count(int64_t start, int64_t end) {
    const char *function_name = "cnp_busday_count";
    if (start == CNP_DATETIME_NAT || end == CNP_DATETIME_NAT) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "NaT is not a business date");
        return 0;
    }
    if (start == end) return 0;
    int64_t result;
    if (start < end) {
        if (busday_count_forward(start, end, &result)) return result;
    } else if (busday_count_forward(end, start, &result)) {
        int64_t reverse = -result;
        if (is_business_day_raw(end)) ++reverse;
        if (is_business_day_raw(start)) --reverse;
        return reverse;
    }
    cnp_set_error(CNP_ERR_VALUE, function_name, "business-day count overflows int64");
    return 0;
}

CNP_API int64_t CNP_CALL cnp_busday_offset(int64_t dt, int64_t offset) {
    const char *function_name = "cnp_busday_offset";
    if (!is_business_day_raw(dt)) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "input date must be a business day");
        return 0;
    }
    if (offset == 0) return dt;
    bool negative = offset < 0;
    uint64_t magnitude = negative
        ? (uint64_t)(-(offset + 1)) + UINT64_C(1) : (uint64_t)offset;
    uint64_t weeks = magnitude / 5;
    uint64_t remainder = magnitude % 5;
    if (weeks > (uint64_t)INT64_MAX / 7) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "business-day offset overflows int64");
        return 0;
    }
    int64_t shift = (int64_t)(weeks * 7);
    if (negative) shift = -shift;
    int64_t result;
    if (!checked_add_i64(dt, shift, &result)) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "business-day offset overflows int64");
        return 0;
    }
    int64_t step = negative ? -1 : 1;
    while (remainder > 0) {
        if (!checked_add_i64(result, step, &result)) {
            cnp_set_error(CNP_ERR_VALUE, function_name, "business-day offset overflows int64");
            return 0;
        }
        if (is_business_day_raw(result)) --remainder;
    }
    return result;
}

CNP_API const char* CNP_CALL cnp_datetime_unit_name(CNP_DATETIME_UNIT unit) {
    if (!require_datetime_unit(unit, "cnp_datetime_unit_name")) return NULL;
    return g_datetime_unit_names[unit];
}

CNP_API CnpArray* CNP_CALL cnp_datetime64_array_create(
    int ndim, const int64_t *shape,
    const int64_t *values, CNP_DATETIME_UNIT unit) {
    const char *function_name = "cnp_datetime64_array_create";
    if (!require_datetime_unit(unit, function_name)) return NULL;
    CnpArray *result = cnp_array_new(
        ndim, shape, CNP_DATETIME, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    if (result->size > 0 && !values) {
        cnp_array_decref(result);
        cnp_set_error(CNP_ERR_VALUE, function_name, "values are required for a non-empty array");
        return NULL;
    }
    if (result->size > 0)
        memcpy(result->data, values, (size_t)result->size * sizeof(int64_t));
    return result;
}

CNP_API CnpArray* CNP_CALL cnp_arange_datetime(
    int64_t start, int64_t stop, int64_t step, CNP_DATETIME_UNIT unit) {
    const char *function_name = "cnp_arange_datetime";
    if (!require_datetime_unit(unit, function_name)) return NULL;
    if (step == 0) {
        cnp_set_error(CNP_ERR_VALUE, function_name, "step must be nonzero");
        return NULL;
    }
    uint64_t count = 0;
    if (step > 0 && start < stop) {
        uint64_t distance = (uint64_t)stop - (uint64_t)start;
        count = (distance - 1) / (uint64_t)step + 1;
    } else if (step < 0 && start > stop) {
        uint64_t distance = (uint64_t)start - (uint64_t)stop;
        uint64_t magnitude = (uint64_t)(-(step + 1)) + UINT64_C(1);
        count = (distance - 1) / magnitude + 1;
    }
    if (count > INT64_MAX) {
        cnp_set_error(CNP_ERR_MEMORY, function_name, "datetime range is too large");
        return NULL;
    }
    int64_t shape[1] = {(int64_t)count};
    CnpArray *result = cnp_array_new(
        1, shape, CNP_DATETIME, CNP_ORDER_C);
    if (!result) {
        cnp_relabel_error(function_name);
        return NULL;
    }
    int64_t *data = (int64_t*)result->data;
    uint64_t value = (uint64_t)start;
    for (uint64_t index = 0; index < count; ++index) {
        data[index] = int64_from_bits(value);
        value += (uint64_t)step;
    }
    return result;
}

static const int64_t *datetime_array_flat_pointer(
    const CnpArray *array, int64_t flat_index, const char *function_name) {
    int64_t coordinates[CNP_MAXDIMS] = {0};
    int64_t remaining = flat_index;
    for (int axis = array->ndim - 1; axis >= 0; --axis) {
        int64_t dimension = array->shape[axis];
        coordinates[axis] = remaining % dimension;
        remaining /= dimension;
    }
    void *address = cnp_array_at(array, coordinates);
    if (!address) cnp_relabel_error(function_name);
    return (const int64_t*)address;
}

CNP_API CNP_STATUS CNP_CALL cnp_datetime_as_string_v2(
    const CnpArray *arr, CNP_DATETIME_UNIT unit,
    char **outputs, int64_t capacity) {
    const char *function_name = "cnp_datetime_as_string_v2";
    if (capacity < 0 || (capacity > 0 && !outputs)) {
        cnp_set_error(
            CNP_ERR_VALUE, function_name,
            "outputs and a non-negative capacity are required");
        return CNP_ERR_VALUE;
    }
    for (int64_t index = 0; index < capacity; ++index) outputs[index] = NULL;
    if (!require_datetime_unit(unit, function_name)) return CNP_ERR_VALUE;
    if (!arr || !arr->dtype || arr->dtype->type_num != CNP_DATETIME) {
        cnp_set_error(CNP_ERR_TYPE, function_name, "a datetime64 array is required");
        return CNP_ERR_TYPE;
    }
    if (capacity < arr->size) {
        cnp_set_error(
            CNP_ERR_SHAPE, function_name,
            "output capacity %lld is smaller than array size %lld",
            (long long)capacity, (long long)arr->size);
        return CNP_ERR_SHAPE;
    }
    for (int64_t index = 0; index < arr->size; ++index) {
        const int64_t *value = datetime_array_flat_pointer(
            arr, index, function_name);
        if (!value) goto fail;
        outputs[index] = cnp_datetime64_to_string(*value, unit);
        if (!outputs[index]) {
            cnp_relabel_error(function_name);
            goto fail;
        }
    }
    return CNP_OK;

fail:
    for (int64_t index = 0; index < capacity; ++index) {
        if (outputs[index]) {
            cnp_char_free_string(outputs[index]);
            outputs[index] = NULL;
        }
    }
    return cnp_get_error(NULL);
}
