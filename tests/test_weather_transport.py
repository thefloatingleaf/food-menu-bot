import io
import ssl
import unittest
from unittest.mock import patch
from urllib.error import URLError

import generate_menu


class WeatherTransportTests(unittest.TestCase):
    def test_verified_response_preserves_utf8_decoding(self) -> None:
        with patch("generate_menu.urlopen", return_value=io.BytesIO("वर्षा".encode())) as open_url:
            result = generate_menu.fetch_url_text("https://weather.example/", timeout=12, retries=0)

        self.assertEqual(result, "वर्षा")
        open_url.assert_called_once_with("https://weather.example/", timeout=12)

    def test_certificate_failures_never_retry_without_verification(self) -> None:
        failures = [
            URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")),
            URLError(ssl.CertificateError("hostname mismatch")),
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                with (
                    patch("generate_menu.urlopen", side_effect=failure) as open_url,
                    patch("generate_menu.time.sleep"),
                    patch("ssl._create_unverified_context") as unverified,
                ):
                    result = generate_menu.fetch_url_text("https://weather.example/", timeout=12, retries=2)

                self.assertIsNone(result)
                self.assertEqual(open_url.call_count, 3)
                for call in open_url.call_args_list:
                    self.assertNotIn("context", call.kwargs)
                unverified.assert_not_called()

    def test_transient_failure_preserves_retry_and_backoff(self) -> None:
        with (
            patch("generate_menu.urlopen", side_effect=[URLError("unavailable"), io.BytesIO(b"forecast")]),
            patch("generate_menu.time.sleep") as sleep,
        ):
            result = generate_menu.fetch_url_text("https://weather.example/", timeout=12, retries=2)

        self.assertEqual(result, "forecast")
        sleep.assert_called_once_with(0.35)

    def test_rejected_imd_certificate_falls_through_to_verified_open_meteo(self) -> None:
        payload = b'{"daily":{"temperature_2m_max":[25],"precipitation_probability_max":[10]},"hourly":{"time":["2026-10-21T08:00"],"temperature_2m":[20]}}'
        config = {"weather_source_order": ["imd", "open_meteo"], "weather_lat": 30, "weather_lon": 75}
        responses = [URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED"))] * 3
        with (
            patch("generate_menu.urlopen", side_effect=responses + [io.BytesIO(payload)]) as open_url,
            patch("generate_menu.time.sleep"),
        ):
            result = generate_menu.resolve_weather_info("2026-10-21", config, generate_menu.parse_weather_thresholds(config))

        self.assertEqual(result.source_hi, "Open-Meteo")
        self.assertEqual(result.morning_temp_c, 20)
        self.assertEqual(open_url.call_count, 4)
        self.assertEqual(open_url.call_args.kwargs, {"timeout": 12})

    def test_all_sources_rejecting_certificates_leave_weather_unavailable(self) -> None:
        config = {"weather_source_order": ["imd", "open_meteo"], "weather_lat": 30, "weather_lon": 75}
        with (
            patch("generate_menu.urlopen", side_effect=URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED"))),
            patch("generate_menu.time.sleep"),
        ):
            result = generate_menu.resolve_weather_info("2026-10-21", config, generate_menu.parse_weather_thresholds(config))

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
