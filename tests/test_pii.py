import pytest

from raglaw.monitoring.pii import StreamRedactor, is_national_id, redact

pytestmark = pytest.mark.unit

ID = "29801011234567"  # 1998-01-01, Dakahlia (12)
ID_AR = "٢٩٨٠١٠١١٢٣٤٥٦٧"
CLEAN = (
    "سن الرشد هي إحدى وعشرون سنة ميلادية كاملة وفقًا للمادة ٤٤، ويتقادم الالتزام "
    "بانقضاء خمس عشرة سنة (المادة ٣٧٤)، وقد ألغيت المواد من ٥٤ إلى ٨٠ في ١٩٥٦/١١/٣."
)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (f"رقمي القومي {ID} فهل يحق لي الطعن؟", "national_id"),
        (f"رقمي القومي {ID_AR} فهل يحق لي الطعن؟", "national_id"),
        ("اتصل بي على 01012345678 من فضلك", "phone"),
        ("My number is +20 101 234 5678.", "phone"),
        ("رقمي ٠١١٢٣٤٥٦٧٨٩ للتواصل", "phone"),
        ("Write to mona.ali@example.com please.", "email"),
        ("هل يحق للسيد أحمد محمد علي أن يطالب بالتعويض؟", "name"),
    ],
)
def test_u42_personal_data_is_redacted(text, kind):
    """U42: a synthetic national ID, a phone number, an email and a name."""
    result = redact(text)

    assert [f.kind for f in result.findings] == [kind]
    assert f"[{kind}]" in result.text
    assert not any(c.isdigit() for c in result.text.split(f"[{kind}]")[0][-3:])


def test_u42_a_clean_arabic_legal_answer_is_unchanged():
    result = redact(CLEAN)

    assert (result.text, result.redacted) == (CLEAN, False)


@pytest.mark.parametrize(
    "digits",
    [
        "12345678901234",  # century digit 1
        "29813011234567",  # month 13
        "29801019934567",  # governorate 99
    ],
)
def test_fourteen_digits_without_an_ids_structure_are_left(digits):
    assert not is_national_id(digits)
    assert redact(f"Case {digits} was filed.").text == f"Case {digits} was filed."


def test_a_streamed_answer_is_redacted_even_when_split_across_pieces():
    text = (
        f"Your ID {ID} and phone 010 1234 5678 are not needed here, as Article 44 says."
    )
    pieces = [text[i : i + 3] for i in range(0, len(text), 3)]  # cuts inside both
    stream = StreamRedactor()

    sent = "".join(stream.feed(p) for p in pieces) + stream.flush()

    assert sent == redact(text).text
    assert "[national_id]" in sent and "[phone]" in sent
    assert [f.kind for f in stream.findings] == ["national_id", "phone"]


def test_a_role_after_an_honorific_is_not_a_name():
    """The Code's only matches before this rule: Articles 651-660's architect."""
    text = "يضمن المهندس المعماري والمقاول متضامنين ما يحدث خلال عشر سنوات"

    assert redact(text).text == text
