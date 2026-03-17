import { z } from "zod";

const MINIMUM_AGE = 1;
const MAXIMUM_AGE = 120;
const emailRegex =
  /^(?!.*\.\.)([A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)@([A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+)$/;

export type CountryPhoneOption = {
  code: string;
  name: string;
  dialCode: string;
};

const priorityCountries: CountryPhoneOption[] = [
  { code: "IN", name: "India", dialCode: "+91" },
  { code: "US", name: "United States", dialCode: "+1" },
  { code: "GB", name: "United Kingdom", dialCode: "+44" },
];

const remainingCountries: CountryPhoneOption[] = [
  ["AF", "Afghanistan", "+93"], ["AL", "Albania", "+355"], ["DZ", "Algeria", "+213"], ["AD", "Andorra", "+376"],
  ["AO", "Angola", "+244"], ["AG", "Antigua and Barbuda", "+1-268"], ["AR", "Argentina", "+54"], ["AM", "Armenia", "+374"],
  ["AU", "Australia", "+61"], ["AT", "Austria", "+43"], ["AZ", "Azerbaijan", "+994"], ["BS", "Bahamas", "+1-242"],
  ["BH", "Bahrain", "+973"], ["BD", "Bangladesh", "+880"], ["BB", "Barbados", "+1-246"], ["BY", "Belarus", "+375"],
  ["BE", "Belgium", "+32"], ["BZ", "Belize", "+501"], ["BJ", "Benin", "+229"], ["BT", "Bhutan", "+975"],
  ["BO", "Bolivia", "+591"], ["BA", "Bosnia and Herzegovina", "+387"], ["BW", "Botswana", "+267"], ["BR", "Brazil", "+55"],
  ["BN", "Brunei", "+673"], ["BG", "Bulgaria", "+359"], ["BF", "Burkina Faso", "+226"], ["BI", "Burundi", "+257"],
  ["CV", "Cabo Verde", "+238"], ["KH", "Cambodia", "+855"], ["CM", "Cameroon", "+237"], ["CA", "Canada", "+1"],
  ["CF", "Central African Republic", "+236"], ["TD", "Chad", "+235"], ["CL", "Chile", "+56"], ["CN", "China", "+86"],
  ["CO", "Colombia", "+57"], ["KM", "Comoros", "+269"], ["CG", "Congo", "+242"], ["CR", "Costa Rica", "+506"],
  ["HR", "Croatia", "+385"], ["CU", "Cuba", "+53"], ["CY", "Cyprus", "+357"], ["CZ", "Czech Republic", "+420"],
  ["CD", "Democratic Republic of the Congo", "+243"], ["DK", "Denmark", "+45"], ["DJ", "Djibouti", "+253"], ["DM", "Dominica", "+1-767"],
  ["DO", "Dominican Republic", "+1-809"], ["EC", "Ecuador", "+593"], ["EG", "Egypt", "+20"], ["SV", "El Salvador", "+503"],
  ["GQ", "Equatorial Guinea", "+240"], ["ER", "Eritrea", "+291"], ["EE", "Estonia", "+372"], ["SZ", "Eswatini", "+268"],
  ["ET", "Ethiopia", "+251"], ["FJ", "Fiji", "+679"], ["FI", "Finland", "+358"], ["FR", "France", "+33"],
  ["GA", "Gabon", "+241"], ["GM", "Gambia", "+220"], ["GE", "Georgia", "+995"], ["DE", "Germany", "+49"],
  ["GH", "Ghana", "+233"], ["GR", "Greece", "+30"], ["GD", "Grenada", "+1-473"], ["GT", "Guatemala", "+502"],
  ["GN", "Guinea", "+224"], ["GW", "Guinea-Bissau", "+245"], ["GY", "Guyana", "+592"], ["HT", "Haiti", "+509"],
  ["HN", "Honduras", "+504"], ["HU", "Hungary", "+36"], ["IS", "Iceland", "+354"], ["ID", "Indonesia", "+62"],
  ["IR", "Iran", "+98"], ["IQ", "Iraq", "+964"], ["IE", "Ireland", "+353"], ["IL", "Israel", "+972"],
  ["IT", "Italy", "+39"], ["JM", "Jamaica", "+1-876"], ["JP", "Japan", "+81"], ["JO", "Jordan", "+962"],
  ["KZ", "Kazakhstan", "+7"], ["KE", "Kenya", "+254"], ["KI", "Kiribati", "+686"], ["KW", "Kuwait", "+965"],
  ["KG", "Kyrgyzstan", "+996"], ["LA", "Laos", "+856"], ["LV", "Latvia", "+371"], ["LB", "Lebanon", "+961"],
  ["LS", "Lesotho", "+266"], ["LR", "Liberia", "+231"], ["LY", "Libya", "+218"], ["LI", "Liechtenstein", "+423"],
  ["LT", "Lithuania", "+370"], ["LU", "Luxembourg", "+352"], ["MG", "Madagascar", "+261"], ["MW", "Malawi", "+265"],
  ["MY", "Malaysia", "+60"], ["MV", "Maldives", "+960"], ["ML", "Mali", "+223"], ["MT", "Malta", "+356"],
  ["MH", "Marshall Islands", "+692"], ["MR", "Mauritania", "+222"], ["MU", "Mauritius", "+230"], ["MX", "Mexico", "+52"],
  ["FM", "Micronesia", "+691"], ["MD", "Moldova", "+373"], ["MC", "Monaco", "+377"], ["MN", "Mongolia", "+976"],
  ["ME", "Montenegro", "+382"], ["MA", "Morocco", "+212"], ["MZ", "Mozambique", "+258"], ["MM", "Myanmar", "+95"],
  ["NA", "Namibia", "+264"], ["NR", "Nauru", "+674"], ["NP", "Nepal", "+977"], ["NL", "Netherlands", "+31"],
  ["NZ", "New Zealand", "+64"], ["NI", "Nicaragua", "+505"], ["NE", "Niger", "+227"], ["NG", "Nigeria", "+234"],
  ["KP", "North Korea", "+850"], ["MK", "North Macedonia", "+389"], ["NO", "Norway", "+47"], ["OM", "Oman", "+968"],
  ["PK", "Pakistan", "+92"], ["PW", "Palau", "+680"], ["PA", "Panama", "+507"], ["PG", "Papua New Guinea", "+675"],
  ["PY", "Paraguay", "+595"], ["PE", "Peru", "+51"], ["PH", "Philippines", "+63"], ["PL", "Poland", "+48"],
  ["PT", "Portugal", "+351"], ["QA", "Qatar", "+974"], ["RO", "Romania", "+40"], ["RU", "Russia", "+7"],
  ["RW", "Rwanda", "+250"], ["KN", "Saint Kitts and Nevis", "+1-869"], ["LC", "Saint Lucia", "+1-758"], ["VC", "Saint Vincent and the Grenadines", "+1-784"],
  ["WS", "Samoa", "+685"], ["SM", "San Marino", "+378"], ["ST", "Sao Tome and Principe", "+239"], ["SA", "Saudi Arabia", "+966"],
  ["SN", "Senegal", "+221"], ["RS", "Serbia", "+381"], ["SC", "Seychelles", "+248"], ["SL", "Sierra Leone", "+232"],
  ["SG", "Singapore", "+65"], ["SK", "Slovakia", "+421"], ["SI", "Slovenia", "+386"], ["SB", "Solomon Islands", "+677"],
  ["SO", "Somalia", "+252"], ["ZA", "South Africa", "+27"], ["KR", "South Korea", "+82"], ["SS", "South Sudan", "+211"],
  ["ES", "Spain", "+34"], ["LK", "Sri Lanka", "+94"], ["SD", "Sudan", "+249"], ["SR", "Suriname", "+597"],
  ["SE", "Sweden", "+46"], ["CH", "Switzerland", "+41"], ["SY", "Syria", "+963"], ["TW", "Taiwan", "+886"],
  ["TJ", "Tajikistan", "+992"], ["TZ", "Tanzania", "+255"], ["TH", "Thailand", "+66"], ["TL", "Timor-Leste", "+670"],
  ["TG", "Togo", "+228"], ["TO", "Tonga", "+676"], ["TT", "Trinidad and Tobago", "+1-868"], ["TN", "Tunisia", "+216"],
  ["TR", "Turkey", "+90"], ["TM", "Turkmenistan", "+993"], ["TV", "Tuvalu", "+688"], ["UG", "Uganda", "+256"],
  ["UA", "Ukraine", "+380"], ["AE", "United Arab Emirates", "+971"], ["UY", "Uruguay", "+598"], ["UZ", "Uzbekistan", "+998"],
  ["VU", "Vanuatu", "+678"], ["VA", "Vatican City", "+379"], ["VE", "Venezuela", "+58"], ["VN", "Vietnam", "+84"],
  ["YE", "Yemen", "+967"], ["ZM", "Zambia", "+260"], ["ZW", "Zimbabwe", "+263"],
].map(([code, name, dialCode]) => ({ code, name, dialCode })) as CountryPhoneOption[];

export const countryPhoneOptions = [
  ...priorityCountries,
  ...remainingCountries.sort((left, right) => left.name.localeCompare(right.name)),
];

function isRealDate(value: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return false;
  }

  const [year, month, day] = value.split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  );
}

export function deriveAgeFromDateOfBirth(dateOfBirth: string, now = new Date()) {
  if (!isRealDate(dateOfBirth)) {
    return null;
  }

  const [year, month, day] = dateOfBirth.split("-").map(Number);
  let age = now.getUTCFullYear() - year;
  const monthDelta = now.getUTCMonth() + 1 - month;
  const dayDelta = now.getUTCDate() - day;

  if (monthDelta < 0 || (monthDelta === 0 && dayDelta < 0)) {
    age -= 1;
  }

  return age;
}

function validatePhoneByCountry(countryCode: string, localPhoneNumber: string, ctx: z.RefinementCtx) {
  const rawValue = localPhoneNumber.trim();
  const digits = rawValue.replace(/\D/g, "");

  if (!/^\d+$/.test(rawValue)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["localPhoneNumber"],
      message: "Enter digits only in the mobile number.",
    });
    return;
  }

  if (countryCode === "IN") {
    if (digits.length < 10) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["localPhoneNumber"],
        message: "Mobile number cannot be less than 10 digits.",
      });
      return;
    }

    if (digits.length !== 10) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["localPhoneNumber"],
        message: "Mobile number must be exactly 10 digits.",
      });
    }
    return;
  }

  if (!/^\d{4,15}$/.test(digits)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["localPhoneNumber"],
      message: "Enter a valid local phone number for the selected country.",
    });
  }
}

export const identitySchema = z
  .object({
    firstName: z.string().trim().min(1, "Enter the first name.").max(80),
    middleName: z.string().trim().max(80).optional().default(""),
    lastName: z.string().trim().min(1, "Enter the last name.").max(80),
    dateOfBirth: z
      .string()
      .trim()
      .refine(isRealDate, "Enter a valid date of birth in YYYY-MM-DD format."),
    location: z.string().trim().min(2, "Enter the present location.").max(160),
    email: z
      .string()
      .trim()
      .min(6, "Enter a valid email address.")
      .max(254, "Enter a valid email address.")
      .regex(emailRegex, "Enter a valid email address.")
      .refine((value) => {
        const domain = value.split("@")[1] ?? "";
        const extension = domain.split(".").at(-1) ?? "";
        return /^[A-Za-z]{2,}$/.test(extension);
      }, "Enter a valid email address with a proper domain extension."),
    countryCode: z.string().trim().min(2, "Choose a country code."),
    localPhoneNumber: z
      .string()
      .trim()
      .min(4, "Enter the local phone number.")
      .max(20, "Enter a valid local phone number."),
  })
  .superRefine((value, ctx) => {
    const matchedCountry = countryPhoneOptions.find((item) => item.code === value.countryCode);
    if (!matchedCountry) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["countryCode"],
        message: "Choose a valid country code.",
      });
    }

    validatePhoneByCountry(value.countryCode, value.localPhoneNumber, ctx);

    const age = deriveAgeFromDateOfBirth(value.dateOfBirth);
    if (age === null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["dateOfBirth"],
        message: "Enter a valid date of birth.",
      });
      return;
    }

    if (age < MINIMUM_AGE || age > MAXIMUM_AGE) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["dateOfBirth"],
        message: "Date of birth must produce an age between 1 and 120 years.",
      });
    }

    if (new Date(`${value.dateOfBirth}T00:00:00.000Z`) > new Date()) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["dateOfBirth"],
        message: "Date of birth cannot be in the future.",
      });
    }
  });

export const responseSchema = z.object({
  attemptId: z.string().trim().optional(),
  lifetimeOptionId: z.string().trim().min(1),
  presentOptionId: z.string().trim().min(1),
});

const usernameRegex = /^[a-z0-9._-]+$/i;

export const loginSchema = z.object({
  username: z
    .string()
    .trim()
    .min(3, "Enter the username.")
    .max(60, "Username is too long.")
    .regex(usernameRegex, "Use letters, numbers, dots, hyphens, or underscores only."),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters.")
    .max(128, "Password must be at most 128 characters."),
});

export const accountCreationSchema = z.object({
  displayName: z.string().trim().min(2, "Enter the account holder name.").max(120),
  username: z
    .string()
    .trim()
    .min(3, "Enter the username.")
    .max(60, "Username is too long.")
    .regex(usernameRegex, "Use letters, numbers, dots, hyphens, or underscores only."),
  password: z
    .string()
    .min(8, "Password must be at least 8 characters.")
    .max(128, "Password must be at most 128 characters."),
});

export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function normalizePhone(countryCode: string, localPhoneNumber: string): string {
  const country = countryPhoneOptions.find((item) => item.code === countryCode);
  const dial = country?.dialCode.replace(/[^\d+]/g, "") ?? "";
  const localDigits = localPhoneNumber.replace(/\D/g, "");
  return `${dial}|${localDigits}`;
}

export function validateIdentityPayload(payload: unknown) {
  return identitySchema.safeParse(payload);
}

export function validateLoginPayload(payload: unknown) {
  return loginSchema.safeParse(payload);
}

export function validateAccountCreationPayload(payload: unknown) {
  return accountCreationSchema.safeParse(payload);
}
