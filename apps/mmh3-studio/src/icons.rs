use rune::Icon;

macro_rules! stroked {
    ($name:ident, $id:literal, $d:literal) => {
        pub static $name: Icon = Icon {
            name: $id,
            view: 16.0,
            body: concat!(r#"<path d=""#, $d, r#"" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>"#),
        };
    };
}

stroked!(SAVE, "studio-save", "M8 2.5V10M4.75 6.75L8 10L11.25 6.75M3 12.5H13");
stroked!(
    REVEAL,
    "studio-reveal",
    "M2.5 4.5C2.5 3.95 2.95 3.5 3.5 3.5H6.25L7.5 5H12.5C13.05 5 13.5 5.45 13.5 6V11.5C13.5 12.05 13.05 12.5 12.5 12.5H3.5C2.95 12.5 2.5 12.05 2.5 11.5Z"
);
stroked!(
    COPY,
    "studio-copy",
    "M5.5 5.5V3.5C5.5 2.95 5.95 2.5 6.5 2.5H12.5C13.05 2.5 13.5 2.95 13.5 3.5V9.5C13.5 10.05 13.05 10.5 12.5 10.5H10.5M3.5 5.5H9.5C10.05 5.5 10.5 5.95 10.5 6.5V12.5C10.5 13.05 10.05 13.5 9.5 13.5H3.5C2.95 13.5 2.5 13.05 2.5 12.5V6.5C2.5 5.95 2.95 5.5 3.5 5.5Z"
);
stroked!(HASH, "studio-hash", "M6 2.5L5 13.5M11 2.5L10 13.5M3 6H13.5M2.5 10H13");
stroked!(RERUN, "studio-rerun", "M12.75 8A4.75 4.75 0 1 1 11.36 4.64M12.75 2.75V5.25H10.25");
stroked!(
    TRASH,
    "studio-trash",
    "M3 4.5H13M6.5 4.5V3C6.5 2.72 6.72 2.5 7 2.5H9C9.28 2.5 9.5 2.72 9.5 3V4.5M4.5 4.5L5.1 12.6C5.14 13.12 5.57 13.5 6.09 13.5H9.91C10.43 13.5 10.86 13.12 10.9 12.6L11.5 4.5"
);
stroked!(TERMINAL, "studio-terminal", "M2.5 3.5H13.5V12.5H2.5ZM5 6.5L7 8L5 9.5M8.5 10H11");

pub static DICE: Icon = Icon {
    name: "studio-dice",
    view: 16.0,
    body: r#"<rect x="2.5" y="2.5" width="11" height="11" rx="2.5" stroke="currentColor" stroke-width="1.2" fill="none"/><circle cx="5.75" cy="5.75" r="1" fill="currentColor"/><circle cx="10.25" cy="10.25" r="1" fill="currentColor"/><circle cx="8" cy="8" r="1" fill="currentColor"/>"#,
};

pub static SPARKLE: Icon =
    Icon { name: "studio-sparkle", view: 16.0, body: r#"<path d="M8 2L9.3 6.7L14 8L9.3 9.3L8 14L6.7 9.3L2 8L6.7 6.7Z" fill="currentColor"/>"# };

stroked!(ZOOM_IN, "studio-zoom-in", "M7 4.75V9.25M4.75 7H9.25M10.25 10.25L13.25 13.25M12 7A5 5 0 1 1 2 7A5 5 0 1 1 12 7Z");
stroked!(ZOOM_OUT, "studio-zoom-out", "M4.75 7H9.25M10.25 10.25L13.25 13.25M12 7A5 5 0 1 1 2 7A5 5 0 1 1 12 7Z");
stroked!(FIT, "studio-fit", "M2.5 6V2.5H6M10 2.5H13.5V6M13.5 10V13.5H10M6 13.5H2.5V10");
